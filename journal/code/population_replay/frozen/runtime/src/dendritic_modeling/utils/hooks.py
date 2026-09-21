"""Shared utilities for hook lifecycles."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

import torch

HookHandle = torch.utils.hooks.RemovableHandle
ItemT = TypeVar("ItemT")
ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)
ResultT = TypeVar("ResultT")
ForwardHook = Callable[..., object]


def remove_hook_handles(handles: Iterable[HookHandle]) -> None:
    """Remove all registered hook handles."""
    for handle in handles:
        handle.remove()


def register_hook_groups(
    items: Iterable[ItemT],
    register: Callable[[ItemT], Iterable[HookHandle]],
) -> list[HookHandle]:
    """Register grouped hooks for arbitrary items with partial-failure cleanup."""
    handles: list[HookHandle] = []
    try:
        for item in items:
            handles.extend(register(item))
        return handles
    except Exception:
        remove_hook_handles(handles)
        raise


def run_with_forward_hooks(
    *,
    attach: Callable[[], Iterable[HookHandle]],
    remove: Callable[[Iterable[HookHandle]], None],
    body: Callable[[], ResultT],
) -> ResultT:
    """Run ``body`` with attached forward hooks and always remove them."""
    handles = attach()
    try:
        return body()
    finally:
        remove(handles)


def iter_child_modules_of_type(
    container: torch.nn.Module | Iterable[object],
    module_type: type[ModuleT],
) -> Iterator[ModuleT]:
    """Yield immediate child modules matching ``module_type`` in container order."""
    children = (
        container.children() if isinstance(container, torch.nn.Module) else container
    )
    for module in children:
        if isinstance(module, module_type):
            yield module


def iter_named_modules_of_type(
    model: torch.nn.Module,
    module_type: type[ModuleT],
) -> Iterator[tuple[str, ModuleT]]:
    """Yield named submodules matching ``module_type`` in model traversal order."""
    for name, module in model.named_modules():
        if isinstance(module, module_type):
            yield name, module


def iter_named_modules_matching(
    model: torch.nn.Module,
    predicate: Callable[[str, torch.nn.Module], bool],
) -> Iterator[tuple[str, torch.nn.Module]]:
    """Yield named submodules matching ``predicate`` in model traversal order."""
    for name, module in model.named_modules():
        if predicate(name, module):
            yield name, module


def iter_modules_of_type(
    model: torch.nn.Module,
    module_type: type[ModuleT],
) -> Iterator[ModuleT]:
    """Yield submodules matching ``module_type`` in module traversal order."""
    for module in model.modules():
        if isinstance(module, module_type):
            yield module


def iter_modules_matching(
    model: torch.nn.Module,
    predicate: Callable[[torch.nn.Module], bool],
) -> Iterator[torch.nn.Module]:
    """Yield submodules matching ``predicate`` in module traversal order."""
    for module in model.modules():
        if predicate(module):
            yield module


def register_named_forward_hooks(
    model: torch.nn.Module,
    module_type: type[ModuleT],
    hook: ForwardHook,
    *,
    predicate: Callable[[str, ModuleT], bool] | None = None,
    prepare: Callable[[str, ModuleT], None] | None = None,
    with_kwargs: bool = False,
) -> list[HookHandle]:
    """Register one forward hook per matching module with partial-failure cleanup."""
    handles: list[HookHandle] = []
    try:
        for name, module in iter_named_modules_of_type(model, module_type):
            if predicate is not None and not predicate(name, module):
                continue
            if prepare is not None:
                prepare(name, module)
            handles.append(module.register_forward_hook(hook, with_kwargs=with_kwargs))
        return handles
    except Exception:
        remove_hook_handles(handles)
        raise


def register_forward_hook_groups(
    model: torch.nn.Module,
    module_type: type[ModuleT],
    register: Callable[[ModuleT], Iterable[HookHandle]],
    *,
    predicate: Callable[[ModuleT], bool] | None = None,
    prepare: Callable[[ModuleT], None] | None = None,
) -> list[HookHandle]:
    """Register one or more forward hooks per matching module.

    This mirrors ``register_named_forward_hook_groups`` for callers whose hook
    records are keyed by object state rather than module names.
    """
    handles: list[HookHandle] = []
    try:
        for module in iter_modules_of_type(model, module_type):
            if predicate is not None and not predicate(module):
                continue
            if prepare is not None:
                prepare(module)
            handles.extend(register(module))
        return handles
    except Exception:
        remove_hook_handles(handles)
        raise


def register_named_forward_hook_groups(
    model: torch.nn.Module,
    module_type: type[ModuleT],
    register: Callable[[str, ModuleT], Iterable[HookHandle]],
    *,
    predicate: Callable[[str, ModuleT], bool] | None = None,
    prepare: Callable[[str, ModuleT], None] | None = None,
) -> list[HookHandle]:
    """Register one or more forward hooks per matching module.

    The per-module ``register`` callback should return handles for hooks it
    successfully attached. If a later module fails, all earlier handles are
    removed before re-raising.
    """
    handles: list[HookHandle] = []
    try:
        for name, module in iter_named_modules_of_type(model, module_type):
            if predicate is not None and not predicate(name, module):
                continue
            if prepare is not None:
                prepare(name, module)
            handles.extend(register(name, module))
        return handles
    except Exception:
        remove_hook_handles(handles)
        raise


class ForwardHookRemovalMixin:
    """Mixin for analyzers whose hook teardown only removes handles."""

    def remove_forward_hooks(self, handles: Iterable[HookHandle]) -> None:
        """Remove registered forward hooks."""
        remove_hook_handles(handles)


__all__ = [
    "ForwardHookRemovalMixin",
    "HookHandle",
    "iter_child_modules_of_type",
    "iter_modules_matching",
    "iter_modules_of_type",
    "iter_named_modules_matching",
    "iter_named_modules_of_type",
    "register_forward_hook_groups",
    "register_hook_groups",
    "register_named_forward_hook_groups",
    "register_named_forward_hooks",
    "remove_hook_handles",
    "run_with_forward_hooks",
]
