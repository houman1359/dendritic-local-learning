"""Shared activation-alias resolution helpers."""

from __future__ import annotations


def resolve_dendritic_activation(
    dendritic_activation,
    reactivate: bool,
    reactivation_type: str | None,
) -> tuple[bool, str]:
    """Resolve paper-facing dendritic activation aliases.

    ``dendritic_activation`` is an alias surface for configs.  When it is
    provided, it takes precedence over ``reactivate``/``reactivation_type``.
    """
    resolved_type = (
        "param_tanh" if reactivation_type is None else str(reactivation_type)
    )
    if dendritic_activation is None:
        if str(resolved_type).lower() in {"none", "identity"}:
            return False, "none"
        return bool(reactivate), resolved_type

    activation_name = str(dendritic_activation).lower()
    if activation_name in {"none", "identity"}:
        return False, "none"
    return True, activation_name


__all__ = ["resolve_dendritic_activation"]
