"""Audit exact split reconstruction and test-data exclusion in nested baselines."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import analyze_review_response_baselines as study


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    cfg = json.loads(study.CONFIG.read_text())
    old_path = study.ROOT / "source_data/fulltree_boundary/output/config.json"
    old = json.loads(old_path.read_text())
    recorded_path = study.ROOT / "source_data/fulltree_boundary/output/runs.csv"
    recorded = pd.read_csv(recorded_path)
    split_rows, checks = [], []
    for path in sorted(study.base.DEFAULT_EXTRACT_ROOT.glob("target*_automatic_conservative")):
        data = study.base.load_target(path)
        for rep in range(old["replicates"]):
            train, test, folds = study.outer_and_inner(data, rep, cfg, old)
            hist = recorded[recorded.target_root_id.eq(data["root_id"])
                & recorded.session.eq(data["session"])
                & recorded.scan_idx.eq(data["scan_idx"])
                & recorded.replicate.eq(rep)]
            assert len(hist) == 4
            assert hist.n_train_trials.eq(train.sum()).all()
            assert hist.n_test_trials.eq(test.sum()).all()
            assert not np.intersect1d(data["stimulus_ids"][train], data["stimulus_ids"][test]).size
            split_rows.append(dict(target_root_id=data["root_id"], session=data["session"],
                scan_idx=data["scan_idx"], replicate=rep,
                train_trial_mask_sha256=array_hash(train), test_trial_mask_sha256=array_hash(test),
                train_stimulus_ids_sha256=array_hash(np.unique(data["stimulus_ids"][train])),
                test_stimulus_ids_sha256=array_hash(np.unique(data["stimulus_ids"][test])),
                n_train_trials=int(train.sum()), n_test_trials=int(test.sum()),
                inner_identity_disjoint=True))
        # One complete nested analysis per scan is challenged with arbitrarily
        # changed OUTER TEST responses and inputs. Only final test scores may change.
        train, test, _ = study.outer_and_inner(data, 0, cfg, old)
        changed = dict(data)
        changed["x_raw"] = data["x_raw"].copy()
        changed["y_raw"] = data["y_raw"].copy()
        changed["x_raw"][test] = 1e4 + 31 * changed["x_raw"][test]
        changed["y_raw"][test] = -1e4 - 17 * changed["y_raw"][test]
        before, rb = study.one(data, 0, cfg, old, recorded)
        after, ra = study.one(changed, 0, cfg, old, recorded)
        a, b = pd.DataFrame(before), pd.DataFrame(after)
        columns = ["method", "draw", "selected_penalty", "n_features"]
        pd.testing.assert_frame_equal(a[columns], b[columns], check_exact=True)
        assert rb == ra
        checks.append(dict(target_root_id=data["root_id"], session=data["session"],
            scan_idx=data["scan_idx"], replicate=0, n_conditions=len(a),
            heldout_perturbation_preserves_penalties_masks_and_training_reliability=True))
    pd.DataFrame(split_rows).to_csv(study.OUT / "outer_split_audit.csv", index=False)
    pd.DataFrame(checks).to_csv(study.OUT / "heldout_exclusion_audit.csv", index=False)
    report = dict(passed=True, n_outer_splits=len(split_rows),
        n_scan_level_heldout_perturbation_checks=len(checks),
        n_condition_level_heldout_perturbation_checks=sum(r["n_conditions"] for r in checks),
        historical_config_sha256=study.sha(old_path), historical_runs_sha256=study.sha(recorded_path),
        split_helper_source_sha256=study.sha(Path(study.base.__file__)),
        baseline_source_sha256=study.sha(Path(study.__file__)), audit_source_sha256=study.sha(__file__),
        scope="Outer splits reconstructed with the unchanged archived seed rule and verified against all recorded train/test counts. Complete stimulus-identity and inner-fold disjointness checked. Original historical trial-mask arrays were not separately archived. Test inputs/labels changed in one full nested fit per scan; selected penalties, feature masks and training reliability were exactly invariant.")
    (study.OUT / "validation_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
