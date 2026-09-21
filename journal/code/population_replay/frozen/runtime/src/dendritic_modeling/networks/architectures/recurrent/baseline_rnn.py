"""
Baseline RNN wrappers (GRU, LSTM, Vanilla RNN) with same interface as EINetwork.
"""

import math
from dataclasses import dataclass, field
from typing import Union

import torch
import torch.nn as nn

from dendritic_modeling.networks.base import BaseNetwork


@dataclass
class BaselineRNNConfig:
    """Configuration for baseline RNN models."""

    cell_type: str = "gru"  # "gru", "lstm", "vanilla"
    input_dim: int = 8
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    output_mode: str = "last"  # "last", "mean", "all"
    input_projection_dims: list[int] = field(default_factory=list)
    use_layernorm: bool = False
    # Initialization for the recurrent weight matrices.
    # "default": PyTorch defaults (uniform with k = 1/sqrt(hidden_dim)).
    # "orthogonal": orthogonal init for the hidden-to-hidden recurrent kernel
    #   (standard fix for vanilla-RNN training stability on long sequences).
    # We default to "default" so that existing training configs reproduce
    # bit-for-bit; new configs that want better long-sequence trainability
    # should explicitly set init_method="orthogonal".
    init_method: str = "default"  # "default" | "orthogonal"
    # LSTM-only: extra bias added to the forget gate at initialization on top
    # of PyTorch's defaults. PyTorch's nn.LSTM initializes the bias to small
    # uniform values, putting the forget gate near sigmoid(0)=0.5 and causing
    # severe cell-state decay over long sequences. The standard convention
    # since Jozefowicz et al. (2015) is +1.0. We default to 0.0 for
    # backward compatibility with existing checkpoints/configs; new configs
    # should explicitly set lstm_forget_bias=1.0.
    lstm_forget_bias: float = 0.0
    # GRU-only: exact effective update-gate biases used at initialization. A
    # positive update bias favors carrying the previous hidden state because
    # PyTorch uses h_t = (1 - z_t) * n_t + z_t * h_{t-1}. The final
    # round(hidden_dim * gru_slow_fraction) coordinates form a deterministic
    # slow group; the remaining coordinates form the fast group. When any of
    # these options is nondefault, bias_hh is zeroed for the update gate and
    # bias_ih is set to these targets. All-zero defaults preserve PyTorch's
    # initialization exactly rather than activating target initialization.
    gru_fast_update_bias: float = 0.0
    gru_slow_update_bias: float = 0.0
    gru_slow_fraction: float = 0.0


class BaselineRNN(BaseNetwork):
    """Wraps nn.GRU, nn.LSTM, or nn.RNN for baseline comparison.

    Provides the same interface as EINetwork:
    - ``is_recurrent`` property
    - ``output_dim`` attribute
    - ``forward(x, hidden=None, return_hidden=False)``
    """

    def __init__(self, config: BaselineRNNConfig):
        super().__init__()
        self.config = config
        self._output_mode = config.output_mode
        self._use_layernorm = bool(config.use_layernorm)

        proj_layers = []
        prev_dim = config.input_dim
        for hidden_dim in config.input_projection_dims:
            proj_layers.append(nn.Linear(prev_dim, hidden_dim))
            proj_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.input_projection = (
            nn.Sequential(*proj_layers) if proj_layers else nn.Identity()
        )
        self.input_layernorm = (
            nn.LayerNorm(prev_dim) if self._use_layernorm else nn.Identity()
        )

        rnn_cls = {
            "gru": nn.GRU,
            "lstm": nn.LSTM,
            "vanilla": nn.RNN,
        }
        cell_type = config.cell_type.lower()
        if cell_type not in rnn_cls:
            raise ValueError(
                f"Unknown cell_type '{config.cell_type}'. "
                f"Choose from: gru, lstm, vanilla"
            )

        self.rnn = rnn_cls[cell_type](
            input_size=prev_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        # Apply opt-in recurrent initialization on top of PyTorch defaults.
        # These are no-ops under the configuration defaults, so existing
        # training runs that did not set them are unaffected.
        self._apply_recurrent_init(
            cell_type,
            init_method=str(config.init_method).lower(),
            lstm_forget_bias=float(config.lstm_forget_bias),
            gru_fast_update_bias=float(config.gru_fast_update_bias),
            gru_slow_update_bias=float(config.gru_slow_update_bias),
            gru_slow_fraction=float(config.gru_slow_fraction),
        )

        self.output_layernorm = (
            nn.LayerNorm(config.hidden_dim) if self._use_layernorm else nn.Identity()
        )
        self.output_dim = config.hidden_dim

    def _apply_recurrent_init(
        self,
        cell_type: str,
        init_method: str = "default",
        lstm_forget_bias: float = 0.0,
        gru_fast_update_bias: float = 0.0,
        gru_slow_update_bias: float = 0.0,
        gru_slow_fraction: float = 0.0,
    ) -> None:
        """Optionally re-initialize the recurrent kernels for training stability.

        - ``init_method="orthogonal"`` orthogonalizes the hidden-to-hidden
          weight blocks of the underlying nn.GRU/nn.LSTM/nn.RNN. For vanilla
          RNNs in particular this is the standard fix for catastrophic
          gradient vanishing on long sequences (Saxe, McClelland, Ganguli
          2013). For GRU and LSTM, PyTorch's ``weight_hh_*`` is a stack of
          gate kernels; we orthogonalize each gate block individually to
          preserve the gating-projection interpretation.
        - ``lstm_forget_bias != 0`` adds the supplied value to the forget-gate
          slice of every LSTM bias (matches Jozefowicz, Zaremba, Sutskever
          2015). PyTorch's nn.LSTM stacks the four gate biases as
          ``[i, f, g, o]``; with ``hidden_size = H`` the forget-gate slice is
          rows ``H:2H``. ``nn.LSTM`` keeps two bias vectors (``bias_ih_l*``
          and ``bias_hh_l*``); we add the offset to ``bias_ih_l*`` only so
          the effective forget-bias offset is exactly the supplied value.
        - GRU update-gate targets split every layer's hidden coordinates into
          deterministic fast and slow groups. PyTorch stacks GRU gates as
          ``[r, z, n]``. The final ``round(H * gru_slow_fraction)`` update-gate
          entries receive ``gru_slow_update_bias`` and the rest receive
          ``gru_fast_update_bias``. In this opt-in mode, the corresponding
          ``bias_hh_l*`` slice is zeroed and ``bias_ih_l*`` is set to the
          target, so their sum is exactly the configured effective bias.
        """
        if init_method not in ("default", "orthogonal"):
            raise ValueError(
                f"Unknown init_method '{init_method}'. Choose from: default, orthogonal"
            )

        gru_bias_values = {
            "gru_fast_update_bias": gru_fast_update_bias,
            "gru_slow_update_bias": gru_slow_update_bias,
            "gru_slow_fraction": gru_slow_fraction,
        }
        for name, value in gru_bias_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if not 0.0 <= gru_slow_fraction <= 1.0:
            raise ValueError(
                "gru_slow_fraction must be in the inclusive range [0, 1], "
                f"got {gru_slow_fraction}"
            )
        if cell_type != "gru" and any(
            value != 0.0 for value in gru_bias_values.values()
        ):
            raise ValueError(
                "GRU update-bias options require cell_type='gru'; "
                f"got cell_type='{cell_type}'"
            )

        H = self.config.hidden_dim
        n_gates = {"vanilla": 1, "gru": 3, "lstm": 4}[cell_type]

        if init_method == "orthogonal":
            for name, p in self.rnn.named_parameters():
                if "weight_hh" not in name:
                    continue
                # weight_hh has shape (n_gates*H, H); orthogonalize each gate slice
                with torch.no_grad():
                    for g in range(n_gates):
                        nn.init.orthogonal_(p[g * H : (g + 1) * H, :])

        if cell_type == "lstm" and lstm_forget_bias != 0.0:
            with torch.no_grad():
                for name, p in self.rnn.named_parameters():
                    if name.startswith("bias_ih_"):
                        p[H : 2 * H].add_(lstm_forget_bias)

        gru_target_init = any(value != 0.0 for value in gru_bias_values.values())
        if cell_type == "gru" and gru_target_init:
            n_slow = math.floor(H * gru_slow_fraction + 0.5)
            n_fast = H - n_slow
            with torch.no_grad():
                for name, p in self.rnn.named_parameters():
                    if name.startswith("bias_ih_"):
                        update_bias = p[H : 2 * H]
                        update_bias[:n_fast].fill_(gru_fast_update_bias)
                        update_bias[n_fast:].fill_(gru_slow_update_bias)
                    elif name.startswith("bias_hh_"):
                        p[H : 2 * H].zero_()

    @property
    def is_recurrent(self) -> bool:
        return True

    def _reduce_outputs(
        self,
        outputs: torch.Tensor,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._output_mode == "last":
            if seq_lengths is not None:
                B, T, _ = outputs.shape
                idx = (seq_lengths - 1).long().clamp(0, T - 1)
                return outputs[torch.arange(B, device=outputs.device), idx]
            return outputs[:, -1, :]
        if self._output_mode == "mean":
            if seq_lengths is not None:
                _B, T, _ = outputs.shape
                lengths = seq_lengths.to(device=outputs.device).long().clamp(0, T)
                mask = torch.arange(T, device=outputs.device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)
                masked = outputs * mask.unsqueeze(-1).to(dtype=outputs.dtype)
                denom = lengths.clamp_min(1).unsqueeze(1).to(dtype=outputs.dtype)
                return masked.sum(dim=1) / denom
            return outputs.mean(dim=1)
        if self._output_mode == "all":
            if seq_lengths is not None:
                _B, T, _ = outputs.shape
                lengths = seq_lengths.to(device=outputs.device).long().clamp(0, T)
                mask = torch.arange(T, device=outputs.device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)
                return outputs * mask.unsqueeze(-1).to(dtype=outputs.dtype)
            return outputs
        raise ValueError(f"Unknown output_mode: {self._output_mode}")

    def forward(
        self,
        x: torch.Tensor,
        hidden=None,
        return_hidden: bool = False,
        seq_lengths: torch.Tensor | None = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, any]]:
        """Forward pass.

        Args:
            x: Input [batch, seq_len, input_dim].
            hidden: Optional hidden state from previous call.
            return_hidden: If True, return (output, hidden_state) tuple.
            seq_lengths: Optional per-sample sequence lengths [batch].
                When provided and output_mode="last", reads output at position
                seq_lengths[i]-1 instead of the last timestep.

        Returns:
            output: Reduced output based on output_mode.
        """
        # Apply per-timestep projection
        if not isinstance(self.input_projection, nn.Identity):
            B, T, D = x.shape
            x = x.reshape(B * T, D)
            x = self.input_projection(x)
            x = x.reshape(B, T, -1)
        if not isinstance(self.input_layernorm, nn.Identity):
            x = self.input_layernorm(x)

        rnn_out, hidden_out = self.rnn(x, hidden)
        if not isinstance(self.output_layernorm, nn.Identity):
            rnn_out = self.output_layernorm(rnn_out)
        out = self._reduce_outputs(rnn_out, seq_lengths=seq_lengths)

        if return_hidden:
            return out, hidden_out
        return out
