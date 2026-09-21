"""Formatting utilities for sweep analysis output."""

from typing import Any


def format_metric_name(metric: str) -> str:
    """
    Format metric name for display.

    Args:
        metric: Raw metric name

    Returns:
        Formatted display name
    """
    return metric.replace("_", " ").title()


def format_value(value: Any, precision: int = 3) -> str:
    """
    Format a value for display.

    Args:
        value: Value to format
        precision: Decimal places for floats

    Returns:
        Formatted string
    """
    if isinstance(value, float):
        if abs(value) < 0.001:
            return f"{value:.2e}"
        return f"{value:.{precision}f}"
    elif isinstance(value, bool):
        return "Yes" if value else "No"
    elif value is None:
        return "N/A"
    else:
        return str(value)


def format_param_combination(params: dict, separator: str = ", ") -> str:
    """
    Format parameter combination for display.

    Args:
        params: Dictionary of parameter values
        separator: Separator between params

    Returns:
        Formatted string
    """
    parts = [f"{k}={format_value(v)}" for k, v in params.items()]
    return separator.join(parts)
