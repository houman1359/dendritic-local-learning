"""
Parametric activation functions.

This module contains various parametric activation functions with learnable parameters.
"""

import logging
import math

import torch
import torch.nn as nn

from dendritic_modeling.networks.utils.weight_transforms import safe_exp


class ParametricActivation(nn.Module):
    """
    Base class for parametric activation functions.

    Provides common initialization and parameter management functionality for parametric
    activation functions with learnable parameters.
    """

    def __init__(self):
        super().__init__()

    def initialize(self, *args, **kwargs):
        """
        Initialize the activation function parameters.
        Should be implemented by child classes.
        """
        raise NotImplementedError(
            "Initialize method must be implemented by child class"
        )

    def forward(self, x):
        """
        Forward pass of the activation function.
        Should be implemented by child classes.
        """
        raise NotImplementedError("Forward method must be implemented by child class")


class ParametricTanh(ParametricActivation):
    """
    A parametric tanh reactivation function with learnable parameters m and b.
    Both parameters are trainable.
    """

    def __init__(self, output_dim, init_m=1.5, init_b=1.0):
        super().__init__()
        self.log_m = nn.Parameter(torch.empty((output_dim,)), requires_grad=True)
        # self.log_b = nn.Parameter(torch.empty((output_dim,)), requires_grad=True)
        self.b = nn.Parameter(torch.empty((output_dim,)), requires_grad=True)

        nn.init.constant_(self.log_m, math.log(init_m))
        nn.init.constant_(self.b, init_b)

    def forward(self, V):
        m = self.log_m.exp()  # slope
        b = self.b  # midpoint
        return (torch.tanh(m * (V - b)) + 1) / 2

    def initialize(self, slope_val, mid_val):
        self.log_m.data.fill_(math.log(slope_val))
        self.b.data.fill_(mid_val)


class ParametricTanhOnlyM(ParametricActivation):
    """
    A parametric tanh activation function where b is fixed and only m is
    trainable. The fixed value of b is provided via the fixed_b argument.
    """

    def __init__(self, output_dim, init_m=1.5, fixed_b=0.5):
        super().__init__()
        self.log_m = nn.Parameter(torch.empty((output_dim,)), requires_grad=True)
        self.fixed_b = fixed_b  # fixed midpoint value
        with torch.no_grad():
            self.log_m.data.fill_(math.log(init_m))

    def forward(self, V):
        m = self.log_m.exp()  # slope
        b = self.fixed_b  # fixed midpoint
        return (torch.tanh(m * (V - b)) + 1) / 2

    def initialize(self, slope_val, mid_val):
        safe_slope = max(float(slope_val), 1e-8)
        self.log_m.data.fill_(math.log(safe_slope))
        self.fixed_b = float(mid_val)


class ParametricLinearSigmoid(ParametricActivation):
    """
    A parametric activation function that interpolates between linear and sigmoid functions.
    The function is: h(x)*g(x) + f(x)*(1-g(x)) where:
    - h(x) = x (linear function)
    - g(x) = exp(x/2) (exponential weighting)
    - f(x) = 4/(1 + exp(-(mx + 2))) (sigmoid function)

    Parameters
    ----------
    output_dim : int
        Dimension of the output
    init_m : float
        Initial value for the slope parameter
    init_b : float or None
        Initial value for the bias parameter
    trainable : bool
        Whether to train the slope parameter (default: True)
    """

    def __init__(self, output_dim, init_m=4, init_b=None, trainable=True):
        super().__init__()
        self.log_m = nn.Parameter(torch.empty((output_dim,)), requires_grad=trainable)

        # Initialize trainable parameter
        if init_m == 0:
            init_m = 1e-8
        nn.init.constant_(self.log_m, math.log(init_m))

        # Fixed parameters for sigmoid
        if init_b is None:
            self.fixed_b = -2.0
        else:
            self.fixed_b = init_b

    def forward(self, x):
        m = self.log_m.exp()  # slope

        # Calculate the three component functions
        h_x = x  # linear function
        g_x = safe_exp(-3 * x, max_exponent=20.0)  # exponential weighting
        f_x = torch.sigmoid(m * x + self.fixed_b)  # sigmoid

        # Combine using the interpolation formula
        return h_x * g_x + f_x * (1 - g_x)

    def initialize(self, slope_val, mid_val):
        safe_slope = max(float(slope_val), 1e-8)
        self.log_m.data.fill_(math.log(safe_slope))
        self.fixed_b = float(mid_val)


class ParametricLinearTanh(ParametricActivation):
    """
    A parametric activation function that interpolates between linear and tanh functions.
    The function is: h(x)*g(x) + f(x)*(1-g(x)) where:
    - h(x) = x (linear function)
    - g(x) = exp(x/2) (exponential weighting)
    - f(x) = (tanh(mx + b)+1)/2 (positivetanh function)

    Parameters
    ----------
    output_dim : int
        Dimension of the output
    init_m : float
        Initial value for the slope parameter
    init_b : float or None
        Initial value for the bias parameter
    trainable : bool
        Whether to train the slope parameter (default: True)
    """

    def __init__(self, output_dim, init_m=4, init_b=None, trainable=True):
        super().__init__()
        self.log_m = nn.Parameter(torch.empty((output_dim,)), requires_grad=trainable)

        # Initialize trainable parameter
        if init_m == 0:
            init_m = 1e-8
        nn.init.constant_(self.log_m, math.log(init_m))

        # Fixed parameters for sigmoid
        if init_b is None:
            self.fixed_b = -2.0
        else:
            self.fixed_b = init_b

    def forward(self, x):
        m = self.log_m.exp()  # slope

        # Calculate the three component functions
        h_x = x  # linear function
        g_x = safe_exp(-3 * x, max_exponent=20.0)  # exponential weighting
        f_x = (torch.tanh(m * x + self.fixed_b) + 1) / 2  # tanh
        # Combine using the interpolation formula
        return h_x * g_x + f_x * (1 - g_x)

    def initialize(self, slope_val, mid_val):
        safe_slope = max(float(slope_val), 1e-8)
        self.log_m.data.fill_(math.log(safe_slope))
        self.fixed_b = float(mid_val)


class ParametricLinearTanhTransition(ParametricActivation):
    """
    A differentiable activation function operating element-wise for a given output dimension.
    Each element starts linear (scalar slope linear_slope) for x < threshold[i],
    and transitions into a scaled and shifted tanh function for x >= threshold[i],
    saturating at 1.
    The transition threshold and transition slope are potentially learnable parameters for each dimension.

    Requires threshold[i] < 1 / linear_slope for proper behavior (saturation at 1).
    Requires transition_slope[i] > 0.
    """

    def __init__(
        self,
        output_dim: int,
        init_threshold=0.0,
        linear_slope=1.0,
        init_transition_slope=None,
        trainable_threshold=True,
        trainable_transition_slope=True,
    ):
        super().__init__()

        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")
        self.output_dim = output_dim

        if linear_slope <= 0:
            raise ValueError("linear_slope must be positive.")
        # linear_slope remains a scalar shared across dimensions
        self.linear_slope = linear_slope

        # --- Initialize Threshold ---
        max_init_threshold = (1.0 / self.linear_slope) - 1e-6
        threshold_vals = self._initialize_param_val(
            init_threshold, output_dim, "init_threshold"
        )
        # Clamp initial values
        clamped_threshold_vals = torch.clamp(threshold_vals, max=max_init_threshold)
        if torch.any(threshold_vals > max_init_threshold):
            logging.warning(
                f"Some initial thresholds violated constraint threshold < {1.0/self.linear_slope:.4f}. They were clamped."
            )

        if trainable_threshold:
            self.threshold = nn.Parameter(clamped_threshold_vals)
        else:
            self.register_buffer("threshold", clamped_threshold_vals)

        # --- Initialize Transition Slope ---
        if init_transition_slope is None:
            # Default to linear_slope if not provided
            init_transition_slope = self.linear_slope
        transition_slope_vals = self._initialize_param_val(
            init_transition_slope, output_dim, "init_transition_slope"
        )
        # Clamp initial values to be positive
        clamped_transition_slope_vals = torch.clamp(transition_slope_vals, min=1e-9)
        if torch.any(transition_slope_vals <= 0):
            logging.warning(
                "Some initial transition slopes were <= 0. They were clamped to min=1e-9."
            )

        if trainable_transition_slope:
            self.transition_slope = nn.Parameter(clamped_transition_slope_vals)
        else:
            self.register_buffer("transition_slope", clamped_transition_slope_vals)

    def _initialize_param_val(self, init_val, dim, name):
        """Helper to initialize parameter values from scalar or tensor/list."""
        if isinstance(init_val, (int, float)):
            return torch.full((dim,), float(init_val))
        elif isinstance(init_val, torch.Tensor):
            if init_val.shape == (dim,):
                return init_val.clone().float()
            else:
                raise ValueError(
                    f"{name} tensor has wrong shape ({init_val.shape}), expected ({dim},)"
                )
        elif isinstance(init_val, (list, tuple)):
            if len(init_val) == dim:
                return torch.tensor(init_val, dtype=torch.float32)
            else:
                raise ValueError(
                    f"{name} list/tuple has wrong length ({len(init_val)}), expected {dim}"
                )
        else:
            raise TypeError(f"Unsupported type for {name}: {type(init_val)}")

    def forward(self, x):
        # x shape: (batch_size, ..., output_dim) or (output_dim,)
        # parameters shape: (output_dim,) - broadcasting will handle element-wise ops

        # Clamp threshold to ensure 1 - m * threshold > 0
        max_threshold = (1.0 / self.linear_slope) - 1e-6  # scalar
        threshold = torch.clamp(
            self.threshold, max=max_threshold
        )  # shape (output_dim,)

        # Clamp transition_slope to ensure > 0
        transition_slope = torch.clamp(
            self.transition_slope, min=1e-8
        )  # shape (output_dim,)

        # Parameters A, B, D now depend on threshold and transition_slope per dimension
        # Broadcasting applies: (output_dim,) op (output_dim,) -> (output_dim,)
        A = 1.0 - self.linear_slope * threshold
        D = self.linear_slope * threshold
        # Add epsilon for stability when A is close to zero
        B = transition_slope / (A + 1e-8)

        # Apply element-wise logic using broadcasting
        # x: (..., output_dim), threshold: (output_dim,) -> comparison: (..., output_dim)
        linear_part = self.linear_slope * x
        # Tanh part involves ops between x and params.
        # (x - threshold) -> shape (..., output_dim)
        # B * (x - threshold) -> shape (..., output_dim)
        # A * tanh(...) + D -> shape (..., output_dim)
        tanh_part = A * torch.tanh(B * (x - threshold)) + D

        output = torch.where(x < threshold, linear_part, tanh_part)

        return output

    def extra_repr(self):
        is_thresh_trainable = isinstance(self.threshold, nn.Parameter)
        is_slope_trainable = isinstance(self.transition_slope, nn.Parameter)
        return (
            f"output_dim={self.output_dim}, linear_slope={self.linear_slope}, "
            f"threshold=Parameter(shape={tuple(self.threshold.shape)}, trainable={is_thresh_trainable}), "
            f"transition_slope=Parameter(shape={tuple(self.transition_slope.shape)}, trainable={is_slope_trainable})"
        )

    def initialize(self, slope_val, mid_val):
        max_threshold = (1.0 / self.linear_slope) - 1e-6
        threshold_val = min(float(mid_val), max_threshold)
        safe_slope = max(float(slope_val), 1e-8)
        self.threshold.data.fill_(threshold_val)
        self.transition_slope.data.fill_(safe_slope)


__all__ = [
    "ParametricLinearSigmoid",
    "ParametricLinearTanh",
    "ParametricLinearTanhTransition",
    "ParametricTanh",
    "ParametricTanhOnlyM",
]
