"""Keep convergence limitations explicit when displaying complete CIFAR cohorts.

This changes no frozen inferential decisions. Opting into a fixed-budget display
requires consistent flags in both the audit and the complete seed-level table.
"""


def validate_convergence_disclosure(summary, outcomes, *, allow_convergence_flags=False):
    audit = summary["audit"]
    valid = audit.get("convergence_valid") is True
    flags = audit.get("convergence_flags", [])
    if valid:
        if flags:
            raise ValueError("Convergence flag contradicts validated audit")
        if "right_censored" in outcomes and outcomes.right_censored.astype(str).str.lower().eq("true").any():
            raise ValueError("Censored run contradicts validated audit")
        return
    if not allow_convergence_flags:
        raise ValueError("Unvalidated convergence; explicit fixed-budget display required")
    if (audit.get("status") != "complete_with_convergence_flags"
            or not flags or summary["decision"].get("audit_passes") is not False):
        raise ValueError("Inconsistent fixed-budget convergence disclosure")
    if "right_censored" not in outcomes:
        raise ValueError("Missing seed-level convergence flags")
    flagged = outcomes[outcomes.right_censored.astype(str).str.lower().eq("true")]
    if len(flagged) != len(flags) or not len(flagged):
        raise ValueError("Audit and seed-level convergence counts disagree")
    for _, row in flagged.iterrows():
        prefix = f"seed-{int(row.seed)}: {row.condition}:"
        if not any(flag.startswith(prefix) for flag in flags):
            raise ValueError("Audit and seed-level convergence identities disagree")
