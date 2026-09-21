"""LIF soma dynamics with surrogate-gradient spikes."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    """Binary spike with fast-sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, overdrive: torch.Tensor, beta: float) -> torch.Tensor:
        ctx.save_for_backward(overdrive)
        ctx.beta = float(beta)
        return (overdrive >= 0).to(dtype=overdrive.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (overdrive,) = ctx.saved_tensors
        beta = ctx.beta
        grad = 1.0 / (1.0 + beta * overdrive.abs()).pow(2)
        return grad_output * grad, None


class LIFSoma(nn.Module):
    """Leaky integrate-and-fire soma used after dendritic integration."""

    def __init__(
        self,
        *,
        threshold: float = 1.0,
        reset: float = 0.0,
        tau: float = 20.0,
        dt: float = 1.0,
        refractory_steps: int = 0,
        surrogate_beta: float = 10.0,
        output_mode: str = "spikes",
        readout_tau: float = 20.0,
    ):
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if readout_tau <= 0:
            raise ValueError(f"readout_tau must be > 0, got {readout_tau}")
        if refractory_steps < 0:
            raise ValueError(f"refractory_steps must be >= 0, got {refractory_steps}")
        output_mode = str(output_mode).lower()
        if output_mode not in {"spikes", "rate", "membrane"}:
            raise ValueError(
                "spike output_mode must be one of 'spikes', 'rate', or 'membrane', "
                f"got {output_mode!r}"
            )

        self.threshold = float(threshold)
        self.reset = float(reset)
        self.tau = float(tau)
        self.dt = float(dt)
        self.refractory_steps = int(refractory_steps)
        self.surrogate_beta = float(surrogate_beta)
        self.output_mode = output_mode
        self.readout_tau = float(readout_tau)

    @property
    def decay(self) -> float:
        return math.exp(-self.dt / self.tau)

    @property
    def readout_decay(self) -> float:
        return math.exp(-self.dt / self.readout_tau)

    def forward(
        self,
        dendritic_voltage: torch.Tensor,
        membrane: torch.Tensor | None,
        refractory_counter: torch.Tensor | None,
        readout_trace: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if membrane is None:
            membrane = torch.zeros_like(dendritic_voltage)
        if refractory_counter is None:
            refractory_counter = torch.zeros_like(dendritic_voltage)
        if readout_trace is None:
            readout_trace = torch.zeros_like(dendritic_voltage)

        decay = torch.as_tensor(
            self.decay, device=dendritic_voltage.device, dtype=dendritic_voltage.dtype
        )
        candidate = decay * membrane + (1 - decay) * dendritic_voltage

        active = (refractory_counter <= 0).to(dtype=dendritic_voltage.dtype)
        candidate = active * candidate + (1 - active) * self.reset
        spikes = SurrogateSpike.apply(candidate - self.threshold, self.surrogate_beta)
        spikes = spikes * active

        new_membrane = candidate * (1 - spikes.detach()) + self.reset * spikes.detach()
        next_refractory = torch.clamp(refractory_counter - 1, min=0)
        if self.refractory_steps > 0:
            next_refractory = torch.where(
                spikes.detach() > 0,
                torch.full_like(next_refractory, float(self.refractory_steps)),
                next_refractory,
            )

        readout_decay = torch.as_tensor(
            self.readout_decay,
            device=dendritic_voltage.device,
            dtype=dendritic_voltage.dtype,
        )
        new_readout = readout_decay * readout_trace + (1 - readout_decay) * spikes

        if self.output_mode == "membrane":
            output = candidate
        elif self.output_mode == "rate":
            output = new_readout
        else:
            output = spikes

        return output, new_membrane, next_refractory, new_readout, spikes


__all__ = ["LIFSoma", "SurrogateSpike"]
