# Frozen H=4 physical-depth factorial

Status: **complete_pass** (360/360 fits).

## Depth and placement

- Aligned serial-BP D4 minus D3: -1.46 pp (95% paired-seed bootstrap interval -1.74 to -1.16; 0/10 positive; exact two-sided sign-flip P=0.0020; primary-family BH q=0.0039; positive gate=fail).
- Aligned serial-BP D4 minus D1: 25.63 pp (95% paired-seed bootstrap interval 24.71 to 26.40; 10/10 positive; exact two-sided sign-flip P=0.0020; primary-family BH q=0.0039; positive gate=pass).
- Serial-BP D4-minus-D3 depth-by-placement interaction: -0.36 pp (95% paired-seed bootstrap interval -0.70 to 0.04; 2/10 positive; exact two-sided sign-flip P=0.1035; primary-family BH q=0.1346; positive gate=fail).

## Point-neuron and local-credit controls

- Serial minus literal grouped point at aligned D4: 25.60 pp (95% paired-seed bootstrap interval 24.69 to 26.37; 10/10 positive; exact two-sided sign-flip P=0.0020; primary-family BH q=n/a; positive gate=pass).
- Architecture-by-placement interaction at D4: 25.91 pp (95% paired-seed bootstrap interval 25.06 to 26.62; 10/10 positive; exact two-sided sign-flip P=0.0020; primary-family BH q=0.0039; positive gate=pass).
- Shared-LocalCA aligned D4 minus D3: 5.27 pp (95% paired-seed bootstrap interval 4.84 to 5.72; 10/10 positive; exact two-sided sign-flip P=0.0020; primary-family BH q=0.0039; positive gate=pass).
- Path-LocalCA aligned D4 minus D3: 0.60 pp (95% paired-seed bootstrap interval 0.04 to 1.25; 7/10 positive; exact two-sided sign-flip P=0.0859; primary-family BH q=0.1176; positive gate=fail).

## Mechanism specificity

- Raw-additive aligned D4 minus D3: -0.55 pp (95% paired-seed bootstrap interval -2.24 to 1.12; 4/10 positive; exact two-sided sign-flip P=0.5645; primary-family BH q=n/a; positive gate=fail).
- Shunting-minus-additive interaction in D4 minus D3: -0.90 pp (95% paired-seed bootstrap interval -2.65 to 0.95; 4/10 positive; exact two-sided sign-flip P=0.3438; primary-family BH q=0.3886; positive gate=fail).

All frozen contrasts, including null or reversed outcomes, remain in the source-data tables. A positive directional statement requires an interval excluding zero and at least 8/10 paired differences in the predicted direction. Exact tests are seed-level; primary-family q values use Benjamini--Hochberg correction.
