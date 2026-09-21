from typing import Union

from torch import nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim: Union[int, tuple[int, ...]]

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in subclass.")

    def get_output_dim(self):
        raise NotImplementedError(
            "get_output_dim method must be implemented in subclass."
        )
