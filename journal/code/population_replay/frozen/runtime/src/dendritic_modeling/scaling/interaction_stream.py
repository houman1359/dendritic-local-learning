"""Resumable device-local online Boolean samples with fixed analytic teachers."""
import torch


class BooleanStream:
    def __init__(self, packet, seed, *, device="cpu", dtype=torch.float32):
        self.packet = packet
        self.device, self.dtype = torch.device(device), dtype
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.masks = torch.as_tensor(packet["masks"].T.copy(), device=self.device, dtype=dtype)
        self.coefficients = torch.as_tensor(packet["coefficients"], device=self.device, dtype=dtype)
        self.draws = 0

    def labels(self, x):
        # All partial sums are integers <=d; float32 exactly represents these.
        counts = ((1 - x) / 2) @ self.masks
        character = 1 - 2 * counts.remainder(2)
        return (character @ self.coefficients)[:, None]

    def next(self, batch_size):
        if batch_size <= 0:
            raise ValueError("Positive batch size required")
        x = torch.randint(0, 2, (batch_size, self.packet["d"]), generator=self.generator,
                          device=self.device, dtype=torch.int8).to(self.dtype) * 2 - 1
        self.draws += batch_size
        return x, self.labels(x)

    def state_dict(self):
        return {"generator": self.generator.get_state().clone(), "draws": self.draws,
                "device_type": self.device.type}

    def load_state_dict(self, state):
        if state["device_type"] != self.device.type:
            raise ValueError("Cross-device RNG identity is not supported")
        self.generator.set_state(state["generator"])
        self.draws = state["draws"]
