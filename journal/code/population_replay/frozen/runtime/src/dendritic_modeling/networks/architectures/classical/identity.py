from dendritic_modeling.networks.base import BaseNetwork


class Identity(BaseNetwork):
    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        self.output_dim = input_dim

    def forward(self, x):
        return x
