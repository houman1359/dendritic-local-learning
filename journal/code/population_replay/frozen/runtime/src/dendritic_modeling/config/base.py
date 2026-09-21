from dataclasses import asdict

from omegaconf import OmegaConf


class BaseConfig:
    @classmethod
    def load(cls, path: str):
        conf = OmegaConf.load(path)
        return cls(**conf)

    def save(self, path: str):
        OmegaConf.save(config=self, f=path)

    def asdict(self):
        return asdict(self)
