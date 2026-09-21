"""Controlled local fan-in, bias learning, and repeated Boolean sample streams."""
import torch

from .interaction_stream import BooleanStream
from .interaction_models import bank, count_bank
from .interaction_relu_controls import construct as relu_construct


FAMILIES = ("local_relu", "local_glu", "local_tanh", "local_rational")


def construct(task, *, dtype=torch.float32):
    family, s, ceiling = task["family"], task["s"], task["ceiling"]
    kwargs = dict(d=task.get("d", 64), s=s, model_seed=task["model_seed"],
                  support_seed=task["support_seed"], dtype=dtype)
    if family == "local_relu":
        model, inv = relu_construct(family, ceiling, **kwargs)
    elif family in FAMILIES:
        cost = count_bank(family, 1, s) - 1
        model, inv = bank(family, units=(ceiling - 1) // cost, **kwargs)
        inv["budget_ceiling"] = ceiling
    else:
        raise ValueError(family)
    sigma = task["bias_scale"]
    if sigma < 0:
        raise ValueError("Nonnegative bias scale required")
    g = torch.Generator(device="cpu").manual_seed(task["bias_seed"])
    with torch.no_grad():
        if sigma == 0:
            model.hidden.bias.zero_()
        else:
            model.hidden.bias.copy_(sigma * torch.randn(model.hidden.bias.shape, generator=g, dtype=dtype))
    mode = task["mode"]
    if mode != "end_to_end":
        if mode not in {"frozen_readout", "bias_readout"}:
            raise ValueError(mode)
        for p in model.parameters(): p.requires_grad_(False)
        for p in model.readout.parameters(): p.requires_grad_(True)
        if mode == "bias_readout": model.hidden.bias.requires_grad_(True)
    inv.update(bias_scale=sigma, bias_seed=task["bias_seed"],
               optimized_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
               initialization="Hidden normal weights variance 1/s; independent normal biases; original head preserved")
    return model, inv


class CyclingBooleanStream(BooleanStream):
    """Replay a fixed pseudorandom pool without storing its coordinate array.

    Canonical 128-example draws preserve the exact CUDA random stream used by
    fresh training. A reset repeats example positions, including their order.
    This is deterministic cycling, not epoch-wise random reshuffling.
    """
    def __init__(self, packet, seed, *, pool_size, device="cpu", dtype=torch.float32):
        super().__init__(packet, seed, device=device, dtype=dtype)
        if pool_size <= 0 or pool_size % 128:
            raise ValueError("Pool size must be a positive multiple of 128")
        self.pool_size = pool_size
        self.initial_generator_state = self.generator.get_state().clone()

    @property
    def unique_example_positions(self):
        return min(self.draws, self.pool_size)

    def next(self, batch_size):
        if batch_size != 128:
            raise ValueError("Use canonical_batch for larger effective batches")
        if self.draws % self.pool_size == 0:
            self.generator.set_state(self.initial_generator_state)
        return super().next(batch_size)

    def state_dict(self):
        return {**super().state_dict(), "pool_size": self.pool_size,
                "initial_generator_state": self.initial_generator_state.clone()}

    def load_state_dict(self, state):
        if state["pool_size"] != self.pool_size:
            raise ValueError("Pool size mismatch")
        super().load_state_dict(state)
        self.initial_generator_state = state["initial_generator_state"].clone()


def canonical_batch(stream, batch_size):
    if batch_size <= 0 or batch_size % 128:
        raise ValueError("Effective batch size must be a positive multiple of 128")
    if batch_size == 128:
        return stream.next(128)
    chunks = [stream.next(128) for _ in range(batch_size // 128)]
    return tuple(torch.cat([p[i] for p in chunks]) for i in range(2))
