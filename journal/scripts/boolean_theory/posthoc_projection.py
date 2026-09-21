#!/usr/bin/env python3
"""Post hoc target-projection diagnostic, specified after fresh Boolean outcomes.

No new fitting or training. Cross-check every exact Walsh projection energy
against a direct conditional-mean calculation on the full truth table.
"""
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

import audit


def main():
    out = audit.DEFAULT_OUT
    rows = []
    for family in audit.FAMILIES:
        values = audit.truth(family)
        mean = sum(map(Q,values))/16
        variance = sum((Q(v)-mean)**2 for v in values)/16
        coeff = audit.walsh(values)
        for mask in range(1,15):
            subset = tuple(j for j in range(4) if mask & (1<<j))
            walsh_energy = sum(coeff[m]**2 for m in range(1,16) if m & ~mask == 0)
            groups = {}
            for bits,value in zip(audit.BITS,values):
                groups.setdefault(tuple(bits[j] for j in subset),[]).append(Q(value)-mean)
            direct_energy = sum((sum(v)/len(v))**2*len(v) for v in groups.values())/16
            assert walsh_energy == direct_energy
            norm_energy = walsh_energy/variance
            if family == "parity4": assert norm_energy == 0
            if family == "xor_of_ands" and len(subset) == 2: assert norm_energy == Q(1,5)
            rows.append(dict(family=family,subset="".join("abcd"[j] for j in subset),
                subset_size=len(subset),projection_energy_centered_raw_rational=str(walsh_energy),
                projection_energy_normalized_rational=str(norm_energy),
                projection_energy_normalized=float(norm_energy),
                zero_direct_label_term_for_all_subtree_functions=norm_energy == 0,
                independent_conditional_mean_check_exact=True,
                analysis_status="post_hoc_after_fresh_learning_outcomes"))
    path = out/"proper_subtree_projection_energy.csv"
    audit.csv_write(path,rows)
    primary_path = out.parent/"boolean_morphology/primary_contrasts.csv"
    report = dict(status="post_hoc_after_fresh_learning_outcomes_no_new_training",
        motivation="Interpret differing parity and XOR-of-AND broadcast behavior; requested after viewing fresh outcomes",
        full_domain_conditional_expectation_crosschecks=98,
        parity_all_14_proper_subsets_exactly_zero=True,
        xor_of_ands_all_six_pair_projection_energies_exactly="1/5",
        scope="At fixed parameters, proper-subtree eligibility is a function only of its own inputs. Zero projection eliminates its direct population target term, not all indirect learning.",
        exclusions="Does not establish broadcast impossibility, a finite-sample zero, or a grouping obstruction",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        audit_source_sha256=hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest(),
        table_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        fresh_primary_table_already_available_sha256=hashlib.sha256(primary_path.read_bytes()).hexdigest())
    (out/"posthoc_projection_validation.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
