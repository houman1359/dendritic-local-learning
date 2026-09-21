"""Extension registry for architecture builders.

The built-in architectures still live in ``factory.get_architecture`` so their
behavior is unchanged.  This registry provides a clean, low-risk extension
surface for project-local architectures and future gradual migration of the
factory dispatcher.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

ArchitectureBuilder = Callable[..., nn.Module]


@dataclass(frozen=True)
class ArchitectureSpec:
    """Metadata for one registered architecture builder."""

    name: str
    builder: ArchitectureBuilder
    description: str = ""


_ARCHITECTURE_REGISTRY: dict[str, ArchitectureSpec] = {}
_RESERVED_ARCHITECTURE_NAMES: set[str] = set()


def reserve_architecture_names(names: set[str] | frozenset[str] | list[str]) -> None:
    """Reserve built-in architecture names so extensions cannot shadow them."""
    _RESERVED_ARCHITECTURE_NAMES.update(str(name).lower() for name in names)


def register_architecture(
    name: str,
    builder: ArchitectureBuilder,
    *,
    aliases: tuple[str, ...] | list[str] = (),
    description: str = "",
    allow_override: bool = False,
) -> None:
    """Register an architecture builder.

    Builders are called as ``builder(parameters, input_dim=..., suffix_input_dim=...)``.
    Existing names are protected unless ``allow_override=True`` is supplied.
    """
    names = [name, *aliases]
    if not names or any(not str(item).strip() for item in names):
        raise ValueError("architecture name and aliases must be non-empty")
    for raw_name in names:
        key = str(raw_name).lower()
        if key in _RESERVED_ARCHITECTURE_NAMES:
            raise ValueError(
                f"Architecture '{key}' is built in and cannot be registered as an extension"
            )
        if key in _ARCHITECTURE_REGISTRY and not allow_override:
            raise ValueError(f"Architecture '{key}' is already registered")
        _ARCHITECTURE_REGISTRY[key] = ArchitectureSpec(
            name=key,
            builder=builder,
            description=description,
        )


def unregister_architecture(name: str) -> None:
    """Remove a registered architecture name if present."""
    _ARCHITECTURE_REGISTRY.pop(str(name).lower(), None)


def get_registered_architecture_names() -> list[str]:
    """Return registered extension architecture names."""
    return sorted(_ARCHITECTURE_REGISTRY)


def get_architecture_spec(name: str) -> ArchitectureSpec | None:
    """Return the spec for a registered extension architecture."""
    return _ARCHITECTURE_REGISTRY.get(str(name).lower())


def build_registered_architecture(
    name: str,
    parameters: Any,
    *,
    input_dim: int | None = None,
    suffix_input_dim: int | None = None,
) -> nn.Module | None:
    """Build a registered extension architecture, or ``None`` if absent."""
    spec = get_architecture_spec(name)
    if spec is None:
        return None
    return spec.builder(
        parameters,
        input_dim=input_dim,
        suffix_input_dim=suffix_input_dim,
    )


__all__ = [
    "ArchitectureBuilder",
    "ArchitectureSpec",
    "build_registered_architecture",
    "get_architecture_spec",
    "get_registered_architecture_names",
    "register_architecture",
    "reserve_architecture_names",
    "unregister_architecture",
]
