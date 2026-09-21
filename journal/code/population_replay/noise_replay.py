"""Replay one original paired noise-control seed after restoring Source Data."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "release_noise"))
from release_hashes import digest, verify_released_file


def main(args):
    journal = args.journal.resolve()
    source = journal / "source_data/curated_publication/noise_controls_provenance.json"
    original = json.loads(source.read_text())["protocol"]
    if args.seed not in original["seeds"]:
        raise ValueError("Choose an original noise-control seed, 2026092100–2026092119")
    protocol = copy.deepcopy(original)
    links, relocated = [], {}
    for old, expected in original["source_sha256"].items():
        relative = old.split("/journal/", 1)[1]
        path = (journal / relative).resolve()
        if not path.is_relative_to(journal):
            raise ValueError("Unsafe source path")
        result = verify_released_file(path, expected)
        if not result["verified"]:
            raise ValueError(f"Unverified source {relative}: {result['reason']}")
        relocated[str(path)] = digest(path)
        links.append(dict(original_path=old, relative_path=relative,
                          original_sha256=expected, actual_sha256=digest(path),
                          verification=result["reason"]))
    protocol["source_sha256"] = relocated
    if args.smoke:
        protocol["steps"] = 1
        protocol["checkpoints"] = [0, 1]
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ["states", "results"]:
        (args.output / name).mkdir()
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    audit = dict(mode="excluded_smoke" if args.smoke else "existing_seed_replay",
                 original_source_record_sha256=digest(source), source_links=links,
                 changes=["relocated source paths with verified original-to-released identities"]
                         + (["explicit one-update smoke budget"] if args.smoke else []),
                 seed=args.seed, steps=protocol["steps"])
    (args.output / "replay.json").write_text(json.dumps(audit, indent=2) + "\n")
    path = journal / "scripts/review_completion/noise_controls.py"
    spec = importlib.util.spec_from_file_location("frozen_noise_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_seed((args.output, args.seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026092100)
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
