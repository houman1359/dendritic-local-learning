"""
Feedback Alignment (FA) and Direct Feedback Alignment (DFA) training strategies.

Implements FA (Lillicrap et al., 2016) and DFA (Nøkland, 2016) for both
standard MLPs and DendriNet architectures. Fixed random feedback matrices
replace the transpose of forward weights during error backpropagation.
"""

import logging
from functools import partial

import torch
import torch.nn as nn

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
)
from dendritic_modeling.training.strategies.standard import Trainer
from dendritic_modeling.utils.hooks import (
    ForwardHookRemovalMixin,
    iter_modules_of_type,
    register_hook_groups,
    run_with_forward_hooks,
)

logger = logging.getLogger(__name__)


class FeedbackAlignmentTrainer(ForwardHookRemovalMixin, Trainer):
    """
    Training strategy using Feedback Alignment (FA) or Direct Feedback
    Alignment (DFA) for error signal propagation.

    FA: errors propagate backward layer-by-layer through fixed random B_l
        instead of W_l^T.
    DFA: each hidden layer receives the output error directly projected
        through its own fixed random matrix B_l.
    """

    def __init__(self, *args, mode: str = "fa", **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode.lower()
        if self.mode not in ("fa", "dfa"):
            raise ValueError(f"mode must be 'fa' or 'dfa', got '{self.mode}'")
        self.filename_prefix = f"{self.mode}_"
        self.logger_info_prefix = f"[{self.mode.upper()}] "
        self._feedback_initialized = False
        # Populated lazily on first training step
        self._layer_info: list[tuple[nn.Module, str, int, int]] = []
        self._B_fa: dict[int, torch.Tensor] = {}  # FA: B[l] shape (out_dim_l, in_dim_l)
        self._B_dfa: dict[int, torch.Tensor] = (
            {}
        )  # DFA: B[l] shape (out_dim_l, output_dim)

    # ------------------------------------------------------------------
    # Feedback matrix initialisation
    # ------------------------------------------------------------------
    def _initialize_feedback(self, model: BaseModel) -> None:
        """Create fixed random feedback matrices matching layer geometry."""
        if self._feedback_initialized:
            return

        self._layer_info = self._collect_trainable_layers(model)
        n = len(self._layer_info)
        if n == 0:
            logger.warning("No trainable layers found for FA/DFA")
            self._feedback_initialized = True
            return

        output_dim = self._layer_info[-1][2]  # out_dim of last layer

        for idx, (_mod, _pname, out_dim, in_dim) in enumerate(self._layer_info):
            # FA feedback: projects error from out_dim back to in_dim
            # B shape (in_dim, out_dim) so that e_{l-1} = e_l @ B^T has shape [B, in_dim]
            self._B_fa[idx] = torch.randn(in_dim, out_dim, device=self.device) * (
                out_dim**-0.5
            )

            # DFA feedback: projects output error to this layer's out_dim
            # B shape (out_dim, output_dim) so that e_l = e_out @ B^T has shape [B, out_dim]
            if idx < n - 1:
                self._B_dfa[idx] = torch.randn(
                    out_dim, output_dim, device=self.device
                ) * (output_dim**-0.5)

        self._feedback_initialized = True
        logger.info(
            f"Initialized {self.mode.upper()} feedback matrices for "
            f"{n} layers (output_dim={output_dim})"
        )

    # ------------------------------------------------------------------
    # Layer discovery
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_trainable_layers(
        model: BaseModel,
    ) -> list[tuple[nn.Module, str, int, int]]:
        """Return (module, param_name, out_features, in_features) for each
        forward-weight matrix in execution order."""
        layers: list[tuple[nn.Module, str, int, int]] = []
        seen_ids: set[int] = set()

        def _try_add(mod: nn.Module, pname: str) -> None:
            if id(mod) in seen_ids:
                return
            param = getattr(mod, pname, None)
            if param is None:
                return
            seen_ids.add(id(mod))
            # BlockLinear.log_weight has shape (out_features, block_size),
            # but the effective input dimension is in_features = out * block_size
            if FeedbackAlignmentTrainer._is_blocklinear_log_weight(mod, pname):
                out_dim = mod.out_features
                in_dim = mod.in_features  # full input dimension
            else:
                out_dim, in_dim = param.shape[:2]
            layers.append((mod, pname, out_dim, in_dim))

        # Prefer DendriticBranchLayer order if present
        branch_layers = list(iter_modules_of_type(model, DendriticBranchLayer))
        if branch_layers:
            for bl in branch_layers:
                if getattr(bl, "branch_excitation", None) is not None:
                    _try_add(bl.branch_excitation, "pre_w")
                if getattr(bl, "branches_to_output", None) is not None:
                    _try_add(bl.branches_to_output, "log_weight")
                if getattr(bl, "branch_inhibition", None) is not None:
                    _try_add(bl.branch_inhibition, "pre_w")

        # Always include decoder nn.Linear layers (e.g. the final classifier head)
        decoder = getattr(model, "decoder_network", None)
        if decoder is not None:
            for mod in iter_modules_of_type(decoder, nn.Linear):
                _try_add(mod, "weight")

        # Fallback: if no layers found yet, scan all nn.Linear
        if not layers:
            for mod in iter_modules_of_type(model, nn.Linear):
                _try_add(mod, "weight")

        return layers

    # ------------------------------------------------------------------
    # Training loop override
    # ------------------------------------------------------------------
    def _epoch_train(
        self, model: BaseModel, train_loader
    ) -> tuple[float, float, float]:
        model.train()
        self._initialize_feedback(model)

        total_loss = 0.0
        total_base = 0.0

        for batch in train_loader:
            x_batch = batch[0].to(self.device)
            y_batch = batch[1].to(self.device)

            # --- forward with activation caching ---
            activations: dict[int, torch.Tensor] = {}
            y_hat = run_with_forward_hooks(
                attach=partial(self._attach_activation_hooks, activations),
                remove=self.remove_forward_hooks,
                body=partial(model, x_batch),
            )

            # --- loss (for logging only) ---
            loss = self.loss_function(model, x_batch, y_batch)
            total_loss += loss.item()
            total_base += loss.item()

            # --- output error ---
            e_output = self._output_error(y_hat, y_batch)

            # --- assign FA/DFA gradients ---
            self.optimizer.zero_grad()
            if self.mode == "fa":
                self._grad_fa(e_output, activations)
            else:
                self._grad_dfa(e_output, activations)

            if self.grad_clip_value:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), self.grad_clip_value
                )
            self.optimizer.step()

        n = len(train_loader)
        return total_loss / n, total_base / n, 0.0

    # ------------------------------------------------------------------
    # Hook helper
    # ------------------------------------------------------------------
    def _attach_activation_hooks(
        self,
        activations: dict[int, torch.Tensor],
    ) -> list[torch.utils.hooks.RemovableHandle]:
        def _register_layer_hook(layer_entry):
            idx, (mod, *_rest) = layer_entry
            return [mod.register_forward_hook(self._make_hook(activations, idx))]

        return register_hook_groups(
            enumerate(self._layer_info),
            _register_layer_hook,
        )

    @staticmethod
    def _make_hook(store: dict, key: int):
        def hook_fn(_module, inp, _out):
            if isinstance(inp, tuple) and len(inp) > 0:
                store[key] = inp[0].detach()
            elif isinstance(inp, torch.Tensor):
                store[key] = inp.detach()

        return hook_fn

    # ------------------------------------------------------------------
    # Output error
    # ------------------------------------------------------------------
    def _output_error(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute dL/dy_hat.  Shape [batch, output_dim]."""
        loss_name = getattr(self.loss_function, "_loss_name", "").lower()
        B = y_hat.size(0)

        if any(
            k in loss_name for k in ("cross entropy", "nll", "negative log likelihood")
        ):
            probs = torch.exp(y_hat)
            if y.dim() == 1 or (y.dim() == 2 and y.size(1) == 1):
                tgt = torch.zeros_like(probs)
                tgt.scatter_(1, y.view(-1, 1).long(), 1.0)
            else:
                tgt = y.float()
            return (probs - tgt) / B

        # MSE-like fallback
        if y.dim() == 1:
            y = y.unsqueeze(1)
        return (y_hat - y.float()) / B

    # ------------------------------------------------------------------
    # Gradient assignment helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_blocklinear_log_weight(mod: nn.Module, pname: str) -> bool:
        """Return whether a parameter name refers to BlockLinear log-weights."""
        return isinstance(mod, BlockLinear) and pname == "log_weight"

    @staticmethod
    def _assign_grad(mod: nn.Module, pname: str, grad_dense: torch.Tensor) -> None:
        """Assign a dense gradient to a parameter, handling BlockLinear specially.

        For BlockLinear.log_weight (shape [out, block_size]), extract only the
        block-diagonal entries from the dense gradient (shape [out, in_features]).
        """
        param = getattr(mod, pname)
        if FeedbackAlignmentTrainer._is_blocklinear_log_weight(mod, pname):
            # grad_dense is (out_features, in_features)
            # param is (out_features, block_size)
            # Output i uses inputs [i*bs : (i+1)*bs], so extract those entries
            bs = mod.block_size
            grad_w = grad_dense.view(mod.out_features, mod.out_features, bs)
            # Each row i should use its own block: the [i, i, :] slice
            idx = torch.arange(mod.out_features, device=grad_dense.device)
            grad_w = grad_w[idx, idx, :]  # (out_features, block_size)
        else:
            grad_w = grad_dense

        g = grad_w.to(param.dtype)
        param.grad = g if param.grad is None else param.grad + g

    @staticmethod
    def _assign_linear_bias_grad(mod: nn.Module, error: torch.Tensor) -> None:
        """Assign a summed bias gradient for Linear modules that own a bias."""
        if not (isinstance(mod, nn.Linear) and mod.bias is not None):
            return

        grad_bias = error.sum(0).to(mod.bias.dtype)
        mod.bias.grad = (
            grad_bias if mod.bias.grad is None else mod.bias.grad + grad_bias
        )

    # ------------------------------------------------------------------
    # FA gradient assignment
    # ------------------------------------------------------------------
    def _grad_fa(
        self,
        e_output: torch.Tensor,
        activations: dict[int, torch.Tensor],
    ) -> None:
        """Layer-by-layer backward propagation through random B.

        NOTE: FA requires a sequential chain of layers where the output dimension
        of layer l matches the input dimension of layer l+1. Dendritic architectures
        have a tree structure (excitation, inhibition, branch-to-output) that breaks
        this assumption. Use DFA (mode='dfa') for dendritic models.
        """
        e = e_output.detach()
        n = len(self._layer_info)

        for idx in range(n - 1, -1, -1):
            mod, pname, out_dim, _in_dim = self._layer_info[idx]
            h_in = activations.get(idx)
            if h_in is None or e.size(-1) != out_dim:
                if idx > 0:
                    e = e @ self._B_fa[idx].t()
                continue

            # grad_W = e^T h_in   -- shape (out_dim, in_dim)
            grad_dense = e.t() @ h_in
            self._assign_grad(mod, pname, grad_dense)

            self._assign_linear_bias_grad(mod, e)

            # propagate error to previous layer via random B
            if idx > 0:
                e = e @ self._B_fa[idx].t()  # [batch, in_dim]

    # ------------------------------------------------------------------
    # DFA gradient assignment
    # ------------------------------------------------------------------
    def _grad_dfa(
        self,
        e_output: torch.Tensor,
        activations: dict[int, torch.Tensor],
    ) -> None:
        """Direct projection of output error to each layer via random B."""
        e_out = e_output.detach()
        n = len(self._layer_info)

        for idx in range(n):
            mod, pname, out_dim, _in_dim = self._layer_info[idx]
            h_in = activations.get(idx)
            if h_in is None:
                continue

            if idx == n - 1:
                # Output layer: use error directly
                e_local = e_out
            else:
                # Hidden layer: project output error to this layer's out_dim
                e_local = e_out @ self._B_dfa[idx].t()  # [batch, out_dim]

            if e_local.size(-1) != out_dim:
                continue

            grad_dense = e_local.t() @ h_in
            self._assign_grad(mod, pname, grad_dense)

            self._assign_linear_bias_grad(mod, e_local)


class ShuntingFeedbackAlignmentTrainer(FeedbackAlignmentTrainer):
    """FA/DFA with shunting (conductance-based) modulation of feedback signals.

    Instead of using raw projected error ``e_local``, modulates it by the local
    shunting signals ``R_tot * (E_rev - V)`` available at each dendritic layer.
    This gives the random feedback biologically plausible gain control and
    directional information from the conductance-based membrane equation.

    Note: shunting modulation is only applied to sub-modules of shunting
    ``DendriticBranchLayer`` instances that store diagnostics (``_diag_g_tot``,
    ``_diag_numerator``). Additive layers gracefully fall back to unmodulated
    DFA/FA behaviour.
    """

    def __init__(self, *args, e_rev_exc: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.e_rev_exc = e_rev_exc
        self._layer_voltages: dict[int, torch.Tensor] = {}
        self._layer_g_tot: dict[int, torch.Tensor] = {}
        # Mapping from _layer_info index -> branch-layer index (built lazily)
        self._layer_to_bl: dict[int, int] = {}
        self._warned_additive = False

    # ------------------------------------------------------------------
    # Hooks for capturing voltage and conductance from DendriticBranchLayer
    # ------------------------------------------------------------------
    @staticmethod
    def _make_voltage_hook(voltage_store: dict, g_tot_store: dict, key: int):
        """Hook to capture post-shunting voltage and total conductance."""

        def hook_fn(module, _inp, output):
            if hasattr(module, "_diag_g_tot") and module._diag_g_tot is not None:
                g_tot_store[key] = module._diag_g_tot.detach()
            if (
                hasattr(module, "_diag_numerator")
                and module._diag_numerator is not None
            ):
                g_tot = g_tot_store.get(key)
                if g_tot is not None:
                    voltage_store[key] = (
                        module._diag_numerator / (g_tot + 1e-8)
                    ).detach()

        return hook_fn

    def _build_layer_to_bl_map(self, branch_layers: list[DendriticBranchLayer]) -> None:
        """Build mapping from ``_layer_info`` index to ``branch_layers`` index.

        ``_layer_info`` contains sub-modules (excitation, blocklinear, inhibition,
        decoder), while ``branch_layers`` contains the parent DendriticBranchLayer
        modules. This mapping lets us look up the correct voltage/g_tot data when
        modulating error at a given sub-module.
        """
        self._layer_to_bl.clear()
        for idx, (mod, *_rest) in enumerate(self._layer_info):
            for bl_idx, bl in enumerate(branch_layers):
                if (
                    mod is getattr(bl, "branch_excitation", None)
                    or mod is getattr(bl, "branches_to_output", None)
                    or mod is getattr(bl, "branch_inhibition", None)
                ):
                    self._layer_to_bl[idx] = bl_idx
                    break

    def _epoch_train(self, model, train_loader):
        model.train()
        self._initialize_feedback(model)

        # Discover DendriticBranchLayers and build index mapping
        branch_layers = list(iter_modules_of_type(model, DendriticBranchLayer))
        self._build_layer_to_bl_map(branch_layers)

        # Warn once if any branch layers are additive (no shunting diagnostics)
        if not self._warned_additive:
            additive_bls = [bl for bl in branch_layers if not bl.use_shunting]
            if additive_bls:
                logger.warning(
                    "ShuntingFeedbackAlignmentTrainer: %d branch layer(s) use "
                    "additive mode. Shunting modulation will be skipped for those "
                    "layers (falling back to standard %s).",
                    len(additive_bls),
                    self.mode.upper(),
                )
                self._warned_additive = True

        # Enable diagnostic storage on DendriticBranchLayer modules
        for bl in branch_layers:
            bl._store_diagnostics = True

        total_loss = 0.0
        total_base = 0.0

        try:
            for batch in train_loader:
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)

                # --- forward with activation + voltage caching ---
                activations: dict[int, torch.Tensor] = {}
                self._layer_voltages.clear()
                self._layer_g_tot.clear()
                y_hat = run_with_forward_hooks(
                    attach=partial(
                        self._attach_shunting_hooks, activations, branch_layers
                    ),
                    remove=self.remove_forward_hooks,
                    body=partial(model, x_batch),
                )

                # --- loss (for logging only) ---
                loss = self.loss_function(model, x_batch, y_batch)
                total_loss += loss.item()
                total_base += loss.item()

                # --- output error ---
                e_output = self._output_error(y_hat, y_batch)

                # --- assign FA/DFA gradients with shunting modulation ---
                self.optimizer.zero_grad()
                if self.mode == "fa":
                    self._grad_fa_shunting(e_output, activations)
                else:
                    self._grad_dfa_shunting(e_output, activations)

                if self.grad_clip_value:
                    torch.nn.utils.clip_grad_value_(
                        model.parameters(), self.grad_clip_value
                    )
                self.optimizer.step()
        finally:
            # Always disable diagnostic storage, even on exception
            for bl in branch_layers:
                bl._store_diagnostics = False

        n = len(train_loader)
        return total_loss / n, total_base / n, 0.0

    def _attach_shunting_hooks(
        self,
        activations: dict[int, torch.Tensor],
        branch_layers: list[DendriticBranchLayer],
    ) -> list[torch.utils.hooks.RemovableHandle]:
        def _attach_voltage_hooks():
            def _register_branch_hook(branch_entry):
                bl_idx, branch_layer = branch_entry
                return [
                    branch_layer.register_forward_hook(
                        self._make_voltage_hook(
                            self._layer_voltages,
                            self._layer_g_tot,
                            bl_idx,
                        )
                    )
                ]

            return register_hook_groups(
                enumerate(branch_layers),
                _register_branch_hook,
            )

        def _register_group(group_name: str):
            if group_name == "activations":
                return self._attach_activation_hooks(activations)
            return _attach_voltage_hooks()

        return register_hook_groups(
            ("activations", "voltages"),
            _register_group,
        )

    def _modulate_error_with_shunting(
        self, e_local: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Modulate projected error by R_tot * (E_rev - V) if available.

        Uses the ``_layer_to_bl`` mapping to translate from ``_layer_info``
        indices to ``branch_layers`` indices where voltage data is stored.
        Falls back to unmodulated error if no shunting data is available.
        """
        bl_idx = self._layer_to_bl.get(layer_idx)
        if (
            bl_idx is not None
            and bl_idx in self._layer_voltages
            and bl_idx in self._layer_g_tot
        ):
            v_n = self._layer_voltages[bl_idx]
            g_tot = self._layer_g_tot[bl_idx]
            R_tot = 1.0 / (g_tot + 1e-8)
            driving = self.e_rev_exc - v_n
            # Average over spatial dims if needed, keep [B, out] shape
            if R_tot.dim() > e_local.dim():
                R_tot = R_tot.mean(dim=-1)
                driving = driving.mean(dim=-1)
            # Match dimensions (sub-module output may differ from branch output)
            if R_tot.size(-1) == e_local.size(-1):
                e_local = e_local * R_tot * driving
        return e_local

    def _grad_fa_shunting(
        self,
        e_output: torch.Tensor,
        activations: dict[int, torch.Tensor],
    ) -> None:
        """FA with shunting modulation at each layer."""
        e = e_output.detach()
        n = len(self._layer_info)

        for idx in range(n - 1, -1, -1):
            mod, pname, out_dim, _in_dim = self._layer_info[idx]
            h_in = activations.get(idx)
            if h_in is None or e.size(-1) != out_dim:
                if idx > 0:
                    e = e @ self._B_fa[idx].t()
                continue

            # Modulate with shunting signals before computing gradient
            e_mod = self._modulate_error_with_shunting(e, idx)
            grad_dense = e_mod.t() @ h_in
            self._assign_grad(mod, pname, grad_dense)

            self._assign_linear_bias_grad(mod, e_mod)

            if idx > 0:
                e = e @ self._B_fa[idx].t()

    def _grad_dfa_shunting(
        self,
        e_output: torch.Tensor,
        activations: dict[int, torch.Tensor],
    ) -> None:
        """DFA with shunting modulation at each layer."""
        e_out = e_output.detach()
        n = len(self._layer_info)

        for idx in range(n):
            mod, pname, out_dim, _in_dim = self._layer_info[idx]
            h_in = activations.get(idx)
            if h_in is None:
                continue

            if idx == n - 1:
                e_local = e_out
            else:
                e_local = e_out @ self._B_dfa[idx].t()

            if e_local.size(-1) != out_dim:
                continue

            # Modulate with shunting signals
            e_local = self._modulate_error_with_shunting(e_local, idx)
            grad_dense = e_local.t() @ h_in
            self._assign_grad(mod, pname, grad_dense)

            self._assign_linear_bias_grad(mod, e_local)
