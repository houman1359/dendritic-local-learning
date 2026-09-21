"""Fresh-density replication of a previously frozen allocation algorithm."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

if __package__:
    from . import smooth_allocation_campaign as campaign
    from .smooth_block import SmoothBlockTask, TaskConfig
else:
    import smooth_allocation_campaign as campaign
    from smooth_block import SmoothBlockTask, TaskConfig


class TiltedTask(SmoothBlockTask):
    def __init__(self, teacher_index):
        if teacher_index not in range(7):
            raise ValueError("Six mixed teachers and one control are declared")
        self.teacher_index = teacher_index
        self._eta = np.zeros(4)
        condition = "equal_range" if teacher_index == 6 else "mixed_range"
        super().__init__(
            TaskConfig(condition=condition, teacher_seed=2026101001 + teacher_index)
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([self.config.teacher_seed, 31])
        )
        if teacher_index < 6:
            self._intervals *= np.exp(
                rng.uniform(np.log(0.8), np.log(1.25), size=(4, 2))
            )
            self._eta = rng.uniform(-0.8, 0.8, size=4)
        else:
            self._eta.fill(0.5)
        nodes, weights = np.polynomial.legendre.leggauss(512)
        values = self._raw_mixtures((nodes[:, None] + 1) / 2)
        self._means = weights @ values / 2
        self._scales = np.sqrt(weights @ ((values - self._means) ** 2) / 2)

    def _raw_mixtures(self, t):
        low, high = self._intervals.T
        delta = high - low
        logarithm = np.log1p(delta / (low + t))
        return (
            t
            / delta
            * (logarithm + self._eta * (2 - (low + high + 2 * t) * logarithm / delta))
        )

    def specification(self):
        return {
            **super().specification(),
            "teacher_index": self.teacher_index,
            "density_tilt": self._eta.tolist(),
            "density": "rho(z)=(1+eta*(2z-a-b)/(b-a))/(b-a), abs(eta)<=.8; mass1 and strictly positive",
            "target": "Four independently normalized positive linearly tilted conductance mixtures; scalar sum divided by2.",
        }


def configuration():
    return {
        **campaign.default_config(),
        "teacher_indices": list(range(7)),
        "budgets": [4, 8, 12, 16, 24, 32, 48],
        "replicates": [0, 1],
        "scope": "Fixed inherited method; six fresh independently sampled interval/density teachers and one separate equal control. No calibration or new HPO. PrimaryP245.",
    }


def observation_seed(teacher_index, replicate):
    return 2026101101 + 10 * teacher_index + 2 * replicate


def choice(family):
    recipe = (
        "readout_ls_then_joint" if family == "tanh" else "alternating_ls_then_joint"
    )
    return {
        "id": f"inherited_{recipe}_600_quantile",
        "threshold_mode": "quantile",
        "fit": campaign.asdict(
            campaign.FitConfig(recipe=recipe, lbfgs_steps=600, learning_rate=0.5)
        ),
    }


def observations(task, seed, config):
    n = config["local_train_n"]
    anchor = task.sample_inputs(seed * 100 + 1, 1)
    y0 = task.evaluate(anchor)[0]
    raw = task.sample_inputs(seed * 100 + 2, n)
    queries = anchor.repeat(4 * n, 1, 1)
    for block in range(4):
        queries[block * n : (block + 1) * n, block] = raw[:, block]
    labels = task.evaluate(queries)
    data = {
        "anchor_x": anchor.numpy(),
        "anchor_y": y0.numpy(),
        "query_x": queries.numpy(),
        "query_y": labels.numpy(),
        "query_id": np.arange(1, 4 * n + 1),
        "query_block": np.repeat(np.arange(4), n),
        "anchor_query_id": np.array(0),
        "local_x": raw.numpy(),
        "local_y": (labels.reshape(4, n).T - y0).numpy(),
    }
    for name, offset, count in [
        ("selector", 3, config["selector_n"]),
        ("test", 4, config["test_n"]),
    ]:
        x, y = task.sample(seed * 100 + offset, count)
        data[f"{name}_x"], data[f"{name}_y"] = x.numpy(), y.numpy()
    return data


def initialize(root):
    config = configuration()
    campaign.write_json(root / "config.json", config)
    specs, oracles = {}, {}
    for index in config["teacher_indices"]:
        task = TiltedTask(index)
        condition = f"teacher{index}"
        specs[condition] = task.specification()
        oracles[condition] = campaign.certificate_allocations(
            specs[condition], config["budgets"], config["capacity_grid"]
        )
        for replicate in config["replicates"]:
            seed = observation_seed(index, replicate)
            data = observations(task, seed, config)
            path = campaign.data_path(root, "confirmation", condition, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                np.savez_compressed(handle, **data)
            campaign.write_json(
                path.with_suffix(".json"),
                {
                    "sha256": campaign.sha(path),
                    "teacher": index,
                    "seed": seed,
                    "replicate": replicate,
                    "train_scalar_queries": 2049,
                    "selector_labels": 1024,
                    "test_labels": 4096,
                    "total_scalar_labels": 7169,
                },
            )
    campaign.write_json(root / "private_teacher_audit.json", specs)
    campaign.write_json(root / "private_certificate_oracle.json", oracles)
    campaign.write_json(
        root / "initialized.json",
        {
            "utc": campaign.now(),
            "config_sha256": campaign.sha(root / "config.json"),
            "fit_count": 1512,
            "assembled_endpoints": 1470,
            "scalar_labels": 100366,
            "teacher_inference_unit": "Six independently generated mixed teachers; two observation replicates averaged within each teacher. Control separate.",
        },
    )


def run(root, index, family):
    config = json.loads((root / "config.json").read_text())
    frozen = json.loads((root / "frozen_sources.json").read_text())
    for name, digest in frozen.items():
        if campaign.sha(Path(__file__).parent / name) != digest:
            raise ValueError(f"Changed source {name}")
    if (
        campaign.sha(root / "config.json")
        != json.loads((root / "initialized.json").read_text())["config_sha256"]
    ):
        raise ValueError("Changed config")
    start = time.monotonic()
    condition = f"teacher{index}"
    rows = []
    for replicate in config["replicates"]:
        seed = observation_seed(index, replicate)
        data = campaign.read_data(root, "confirmation", condition, seed)
        library = campaign.fit_library(
            root,
            "confirmation",
            condition,
            family,
            choice(family),
            seed,
            data,
            config["capacity_grid"],
        )
        rows.extend(
            campaign.evaluate_policies(
                root,
                "confirmation",
                condition,
                family,
                choice(family),
                seed,
                data,
                library,
                config["budgets"],
                config,
            )
        )
    campaign.write_json(
        root / "confirmation" / f"{condition}_{family}_complete.json",
        {
            "rows": rows,
            "teacher": index,
            "family": family,
            "elapsed_seconds": time.monotonic() - start,
            "utc": campaign.now(),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["initialize", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--teacher", type=int)
    parser.add_argument("--family", choices=["shunt", "relu", "tanh"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "initialize":
        initialize(args.root)
    else:
        if args.teacher not in range(7) or args.family is None:
            parser.error("Declared teacher and family required")
        run(args.root, args.teacher, args.family)


if __name__ == "__main__":
    main()
