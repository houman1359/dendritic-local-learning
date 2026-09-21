"""
gradient_scaling.py
===================
Provides a GradientScaler class that registers backward hooks to implement
various gradient-scaling strategies for BlockLinear or other modules.
"""

import functools
import logging

import torch

logger = logging.getLogger(__name__)
_WARNED_HOOK_FAILURES: set[tuple[str, str, str]] = set()


def _warn_hook_fallback(context: str, exc: Exception) -> None:
    """Warn once when a gradient-scaling hook falls back to unscaled gradients."""
    key = (context, type(exc).__name__, str(exc))
    if key in _WARNED_HOOK_FAILURES:
        return
    _WARNED_HOOK_FAILURES.add(key)
    logger.warning(
        "Gradient-scaling hook %s failed; returning unscaled gradients. "
        "Fix this warning if the run depends on gradient_scaling. Error: %s",
        context,
        exc,
        exc_info=True,
    )


def safe_hook(hook_fn):
    """
    Decorator for backward hooks to ensure they return None when no inputs require gradients.
    This prevents the "Backward hook for Modules where no input requires gradient" error.

    Args:
        hook_fn: The original hook function to decorate

    Returns:
        A decorated hook function that properly handles the case when no inputs require gradients
    """

    @functools.wraps(hook_fn)
    def safe_hook_fn(module, grad_input, grad_output):
        # Check if any grad_input requires gradient
        requires_grad = False
        if grad_input is not None:
            for g in grad_input:
                if g is not None:
                    requires_grad = True
                    break

        # Check parameters too (some hooks need this)
        if not requires_grad:
            for p in module.parameters(recurse=True):
                if p.requires_grad:
                    requires_grad = True
                    break

        # If nothing requires gradients, return None
        if not requires_grad:
            return None

        # Otherwise, call the original hook function
        try:
            result = hook_fn(module, grad_input, grad_output)
            return result
        except Exception as e:
            _warn_hook_fallback(getattr(hook_fn, "__name__", "unknown"), e)
            return grad_input

    return safe_hook_fn


def amp_safe_dtype_cast(grad, target_dtype=torch.float32):
    """
    Safely cast gradient to target dtype for AMP compatibility.

    Args:
        grad: Input gradient tensor
        target_dtype: Target dtype (default: float32)

    Returns:
        Gradient cast to target dtype, or original if None
    """
    if grad is None:
        return None

    # Handle different gradient types and ensure compatibility
    if isinstance(grad, torch.Tensor):
        if grad.dtype != target_dtype:
            return grad.to(target_dtype)

    return grad


def scale_activation_grad_hook(module, grad_input, grad_output):
    """
    A hook for reactivation functions. If inputs don't require gradients,
    return None as required.
    """
    # Check if prints are enabled
    print_hooks = getattr(module, "print_hooks", False)

    # Check for AMP compatibility mode
    disable_hooks_with_amp = getattr(module, "disable_hooks_with_amp", False)
    if (
        disable_hooks_with_amp
        and hasattr(module, "_amp_enabled")
        and module._amp_enabled
    ):
        if print_hooks:
            logger.info("Skipping reactivation hook due to AMP compatibility mode")
        return grad_output

    if print_hooks:
        logger.info("Scaling reactivation gradient")

    # Simply return grad_output to preserve dtypes
    # The actual reactivation gradient scaling should be implemented
    # in the forward pass or through other means that don't change dtypes
    return grad_output


def scale_layer_input_gradients_elementwise(module, grad_input, grad_output):
    """
    Apply element-wise scaling to the input gradients.
    """
    input_scale_vec = 1
    if hasattr(module, "input_scale_vec"):
        if module.input_scale_vec is not None:
            input_scale_vec = module.input_scale_vec

    # Check if prints are enabled
    print_hooks = getattr(module, "print_hooks", False)

    # Check for AMP compatibility mode
    disable_hooks_with_amp = getattr(module, "disable_hooks_with_amp", False)
    if (
        disable_hooks_with_amp
        and hasattr(module, "_amp_enabled")
        and module._amp_enabled
    ):
        if print_hooks:
            logger.info("Skipping input gradient scaling due to AMP compatibility mode")
        return grad_input

    if print_hooks:
        for i, grad in enumerate(grad_input):
            if grad is not None:
                # Use AMP-safe dtype casting for statistics
                grad_float32 = amp_safe_dtype_cast(grad, torch.float32)
                scaled_grad_float32 = grad_float32 * input_scale_vec

                mean_abs_before = grad_float32.abs().mean().item()
                mean_abs_after = scaled_grad_float32.abs().mean().item()
                logger.info(
                    "Gradient %s - Mean Abs Before: %.6f, Mean Abs After: %.6f",
                    i,
                    mean_abs_before,
                    mean_abs_after,
                )

    # Apply scaling while preserving original dtypes
    scaled_gradients = []
    for grad in grad_input:
        if grad is not None:
            # Direct scaling to preserve dtype
            scaled_grad = grad * input_scale_vec
            scaled_gradients.append(scaled_grad)
        else:
            scaled_gradients.append(None)

    return tuple(scaled_gradients)


def scale_layer_param_gradients_elementwise(
    grad,
    param_scale_vec,
    print_hooks=False,
    disable_hooks_with_amp=False,
    amp_enabled=False,
):
    """
    Apply element-wise scaling to the parameter gradients.
    """
    param_scale_vec = 1 if param_scale_vec is None else param_scale_vec

    # Check for AMP compatibility mode
    if disable_hooks_with_amp and amp_enabled:
        if print_hooks:
            logger.info(
                "Skipping parameter gradient scaling due to AMP compatibility mode"
            )
        return grad

    if grad is None:
        return None

    if print_hooks:
        # Use AMP-safe dtype casting for statistics only
        grad_float32 = amp_safe_dtype_cast(grad, torch.float32)
        scaled_grad_float32 = grad_float32 * param_scale_vec

        mean_abs_before = grad_float32.abs().mean().item()
        mean_abs_after = scaled_grad_float32.abs().mean().item()
        logger.info(
            "Parameter Gradient - Mean Abs Before: %.6f, Mean Abs After: %.6f",
            mean_abs_before,
            mean_abs_after,
        )

    # Direct scaling to preserve dtype
    scaled_grad = grad * param_scale_vec
    return scaled_grad


def scale_layer_gradients_scalar(module, grad_input, grad_output, scale_factor):
    """
    Multiply entire grad_input by a single scalar factor.
    """
    # Check if prints are enabled
    print_hooks = getattr(module, "print_hooks", False)

    # Calculate and print mean absolute gradient values before and after scaling
    if print_hooks:
        for i, g in enumerate(grad_input):
            if g is not None:
                mean_abs_before = g.abs().mean().item()
                mean_abs_after = (g * scale_factor).abs().mean().item()
                logger.info(
                    "Scalar Scaling Gradient %s - Mean Abs Before: %.6f, "
                    "Mean Abs After: %.6f (scale=%.4f)",
                    i,
                    mean_abs_before,
                    mean_abs_after,
                    scale_factor,
                )

    return tuple(g * scale_factor if (g is not None) else None for g in grad_input)


# Apply the safe_hook decorator to create safe versions
# safe_scale_layer_input_gradients_elementwise = (
#     lambda module, grad_input, grad_output: safe_hook(
#         lambda m, gi, go: scale_layer_input_gradients_elementwise(m, gi, go)
#     )(module, grad_input, grad_output)
# )

# safe_scale_layer_gradients_scalar = (
#     lambda module, grad_input, grad_output, scale_factor: safe_hook(
#         lambda m, gi, go: scale_layer_gradients_scalar(m, gi, go, scale_factor)
#     )(module, grad_input, grad_output)
# )


class GradientScaler:
    """
    A class for hooking into BlockLinear or other modules to apply
    user-selected gradient scaling strategies.
    """

    def __init__(
        self,
        reactivation_strategy="none",
        blocklinear_strategy="none",
        topk_strategy="none",
        scale_factor=1.0,
        layer_idx=0,
        print_hooks=False,
        disable_hooks_with_amp=False,
        amp_enabled=False,
    ):
        """
        Parameters
        ----------
        strategy : str
            One of ['none', 'distal_upweight_by_idx', 'block_conductance_dynamic'].
        scale_factor : float
            Base scaling factor for certain strategies.
        layer_idx : int
            For 'distal_upweight_by_idx' if needed.
        print_hooks : bool
            Whether to print gradient statistics in hooks. Default is False.
        disable_hooks_with_amp : bool
            Whether to disable hooks when AMP is enabled. Default is False.
        amp_enabled : bool
            Whether AMP is currently enabled. Default is False.
        """
        self.reactivation_strategy = reactivation_strategy
        self.blocklinear_strategy = blocklinear_strategy
        self.topk_strategy = topk_strategy
        self.scale_factor = scale_factor
        self.layer_idx = layer_idx
        self.print_hooks = print_hooks
        self.disable_hooks_with_amp = disable_hooks_with_amp
        self.amp_enabled = amp_enabled

    def register_reactivation_inverse(self, reactivation_layer):
        """
        Register a backward hook to invert the reactivation function gradient.
        """
        if self.reactivation_strategy != "none":
            # First remove any existing hooks to prevent conflicts
            if hasattr(reactivation_layer, "_backward_hooks"):
                reactivation_layer._backward_hooks.clear()
            if hasattr(reactivation_layer, "_backward_pre_hooks"):
                reactivation_layer._backward_pre_hooks.clear()

            # Set the print_hooks attribute
            reactivation_layer.print_hooks = self.print_hooks

            # Set AMP compatibility attributes
            reactivation_layer.disable_hooks_with_amp = self.disable_hooks_with_amp
            reactivation_layer._amp_enabled = self.amp_enabled

            # Register with full backward hook consistently
            reactivation_layer.register_full_backward_hook(
                # safe_hook(scale_activation_grad_hook)
                scale_activation_grad_hook
            )

    def register_block_linear_dynamic(self, block_linear_layer):
        """
        If strategy='conductance_dynamic', attach a backward hook that
        computes conduction-based scaling from the current block plus excit/inhib.

        If 'distal_upweight_by_idx', we do (layer_idx+1) * scale_factor for entire param.
        If 'none', no special scaling.
        """
        if self.print_hooks:
            logger.info(
                "Registering block linear dynamic, strategy: %s",
                self.blocklinear_strategy,
            )

        def input_hook_fn(module, grad_input, grad_output):
            # Apply the appropriate scaling strategy
            if self.print_hooks:
                logger.info("BlockLinear input hook call")
            if self.blocklinear_strategy == "conductance_dynamic":
                if getattr(module, "_use_forward_dynamic_grad_scaling", False):
                    return grad_input
                try:
                    return scale_layer_input_gradients_elementwise(
                        module, grad_input, grad_output
                    )
                except Exception as exc:
                    _warn_hook_fallback("blocklinear_input_conductance_dynamic", exc)
                    return grad_input

            elif self.blocklinear_strategy == "distal_upweight_by_idx":
                try:
                    scale_val = (self.layer_idx + 1) * self.scale_factor
                    return scale_layer_gradients_scalar(
                        module, grad_input, grad_output, scale_val
                    )
                except Exception as exc:
                    _warn_hook_fallback("blocklinear_input_distal_upweight", exc)
                    return grad_input

            else:
                return grad_input

        def param_hook_fn(grad):
            if self.print_hooks:
                logger.info("BlockLinear parameter hook call")
            if self.blocklinear_strategy == "conductance_dynamic":
                if getattr(
                    block_linear_layer, "_use_forward_dynamic_grad_scaling", False
                ):
                    return grad
                try:
                    return scale_layer_param_gradients_elementwise(
                        grad,
                        block_linear_layer.param_scale_vec,
                        self.print_hooks,
                        self.disable_hooks_with_amp,
                        self.amp_enabled,
                    )
                except Exception as exc:
                    _warn_hook_fallback("blocklinear_param_conductance_dynamic", exc)
                    return grad

            else:
                return grad

        if self.blocklinear_strategy != "none":
            # First remove any existing hooks to prevent conflicts
            if hasattr(block_linear_layer, "_backward_hooks"):
                block_linear_layer._backward_hooks.clear()
            if hasattr(block_linear_layer, "_backward_pre_hooks"):
                block_linear_layer._backward_pre_hooks.clear()

            # Set the print_hooks attribute
            block_linear_layer.print_hooks = self.print_hooks

            # Set AMP compatibility attributes
            block_linear_layer.disable_hooks_with_amp = self.disable_hooks_with_amp
            block_linear_layer._amp_enabled = self.amp_enabled

            # Register consistently with full backward hook
            block_linear_layer.register_full_backward_hook(input_hook_fn)
            block_linear_layer.log_weight.register_hook(param_hook_fn)
            if self.print_hooks:
                logger.info("Registered block linear dynamic")

    def register_topk_dynamic(self, topk_layer):
        """
        If strategy='conductance_dynamic', attach a backward hook that
        computes conduction-based scaling from the current block plus excit/inhib.

        If 'distal_upweight_by_idx', we do (layer_idx+1) * scale_factor for entire param.
        If 'none', no special scaling.
        """
        if self.print_hooks:
            logger.info("Registering TopK dynamic, strategy: %s", self.topk_strategy)

        def input_hook_fn(module, grad_input, grad_output):
            # Apply the appropriate scaling strategy
            if self.print_hooks:
                logger.info("TopK input hook call")
            if self.topk_strategy == "conductance_dynamic":
                if getattr(module, "_use_forward_dynamic_grad_scaling", False):
                    return grad_input
                try:
                    return scale_layer_input_gradients_elementwise(
                        module, grad_input, grad_output
                    )
                except Exception as exc:
                    _warn_hook_fallback("topk_input_conductance_dynamic", exc)
                    return grad_input

            elif self.topk_strategy == "distal_upweight_by_idx":
                try:
                    scale_val = (self.layer_idx + 1) * self.scale_factor
                    return scale_layer_gradients_scalar(
                        module, grad_input, grad_output, scale_val
                    )
                except Exception as exc:
                    _warn_hook_fallback("topk_input_distal_upweight", exc)
                    return grad_input

            else:
                return grad_input

        def param_hook_fn(grad):
            if self.print_hooks:
                logger.info("TopK parameter hook call")
            if self.topk_strategy == "conductance_dynamic":
                if getattr(topk_layer, "_use_forward_dynamic_grad_scaling", False):
                    return grad
                try:
                    return scale_layer_param_gradients_elementwise(
                        grad,
                        topk_layer.param_scale_vec,
                        self.print_hooks,
                        self.disable_hooks_with_amp,
                        self.amp_enabled,
                    )
                except Exception as exc:
                    _warn_hook_fallback("topk_param_conductance_dynamic", exc)
                    return grad

            else:
                return grad

        if self.topk_strategy != "none":
            if self.topk_strategy == "conductance_dynamic":
                topk_layer._use_forward_dynamic_grad_scaling = True

            # First remove any existing hooks to prevent conflicts
            if hasattr(topk_layer, "_backward_hooks"):
                topk_layer._backward_hooks.clear()
            if hasattr(topk_layer, "_backward_pre_hooks"):
                topk_layer._backward_pre_hooks.clear()

            # Set the print_hooks attribute
            topk_layer.print_hooks = self.print_hooks

            # Set AMP compatibility attributes
            topk_layer.disable_hooks_with_amp = self.disable_hooks_with_amp
            topk_layer._amp_enabled = self.amp_enabled

            # Register consistently with full backward hook
            # topk_layer.register_full_backward_hook(input_hook_fn)
            topk_layer.pre_w.register_hook(param_hook_fn)


# Apply the safe_hook decorator to our existing functions
# scale_activation_grad_hook = safe_hook(scale_activation_grad_hook)
