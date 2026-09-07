# Final reference, cohort and coordinate review

Read-only review of the expanded SI source, main captions and delivery notation. No manuscript or scientific code was changed.

## Concrete reference corrections

1. In SI S9, the sentence “This is the exact adjoint-replacement contrast plotted in main Fig.~8D” must point to **main Fig. 8C** after the C/D swap. The S48 fragment correctly points to 8D for signed physical calibration and should retain that reference.
2. In main Methods, “The optional four- and five-factor rules add stage-level preconditioners and are described in Supplementary Section~S1” must cite **S2**. The relevant definitions are under “Local rules and feedback objects,” after its three-factor update equation.

The expanded SI contains exactly 52 figure environments in the intended sequence. S48, S49, S50, S51 and S52 correspond to normalized shunt dose, MNIST controls, first conductance study, opposed-conductance robustness and expanded conductance learning rates. S47 uses the nonnumbered asset filename `figure_shunt_weak_channels.pdf`, but occupies the correct S47 counter position. Hardcoded main references to Figures 2, 3, 5, 6, 8 and 9 otherwise match the current narrative. Main Table S9, S13/S14 and S16 references point to the intended full-tree, notation/dictionary and strict-scalar tables.

## Legacy versus fresh MNIST cohorts

These edits become necessary when main Figure 1 changes to the fresh six-rule cohort; retain all old data and counts.

- SI S2, near “We therefore trained a separate 15-seed strict-scalar condition”: replace its ending “and use it, rather than matched-width scalar-fallback feedback, for the scalar rung in the main figure” with “which supplies the scalar rung in the original ladder retained in Supplementary Fig.~S45. The fresh six-rule ladder also uses strict scalar feedback.”
- SI S5: change “The main feedback ladder used 15 seeds (42--56)” to “The original feedback ladder retained in Supplementary Fig.~S45 used 15 seeds (42--56).”
- Table S11: label “Main feedback ladder” as “Original feedback ladder (S45)”; change the fallback row's “Same as the main feedback ladder” to “Same as the original feedback ladder.” The rate values remain unchanged.
- Table S12 footnote: change “the current 15-seed feedback ladder” to “the original 15-seed feedback ladder.”
- Table S15 caption: identify its matched 90-fit accuracy ladder as the **original 15-seed ladder retained in S45**. It does not contain the fresh six-arm outcomes.
- S45 caption: add “original” to its opening description so the explicit fifteen-seed count cannot be mistaken for the new main cohort. S30 already correctly labels its frozen references as the legacy fifteen-seed ladder.
- Table S36 cohort index: label the existing image row “Original image feedback ladders” and add a separate fresh MNIST row after complete audit: six rules, three development seeds per architecture/rule/rate (108 development fits); ten fresh seeds per architecture/rule; selected and common-rate views share coincident fits. The agent currently reports 190 unique fresh fits, but the final audit must supply that count.
- The input `image_ladder_controls_methods` currently appears in S5 inside the physical-depth control discussion, just before `physical_depth_budget_methods`. Move only that image-methods input to the end of “Training protocols and feedback cohorts,” before the CIFAR-10 subsection, so a reader following the image protocol does not need to search through the depth analyses. Its paragraph heading does not alter section numbering.

Root already plans to replace main Figure 1's old six-panel caption/references and fifteen-seed Methods description with the fresh five-panel mapping. The new C/D data and old E controls must remain explicitly separate cohorts; S45 retains the original readout-derived data. The historical voltage-capture values also need to remain labeled as historical checkpoints, distinct from the fresh activation-capture results in S49.

## Coordinate and gradient wording

1. **Sign-sensitive wording:** in `conductance_credit_demand_methods.tex`, change “per-example approximate updates have nonnegative inner products with exact per-example gradients” to “per-example approximate gradient estimates have nonnegative inner products with exact per-example gradients.” A descent update has the opposite sign; the diagnostic precedes optimizer transformation.
2. In main Methods, the context-cancellation definition should use “gradient estimates before optimizer transformation” rather than an unspecified “mean update.” This matches the corrected Results and avoids implying an audit of Adam steps or momentum.
3. The SI opening still defines $K$ as “the number of communicated feedback channels.” Prefer “the number of fixed spatial delivery profiles; it does not count independent external error signals.” The later S2 explanation and Table S13 already make this distinction correctly.
4. In Table S13, add the activation-space counterpart to the address row: “The same address matrix can act on activation-space errors; fresh MNIST projections use $\widehat{\boldsymbol\delta}^{a}_u=A_u\boldsymbol c_u$ and retain activation derivatives in eligibility.” The existing voltage-space equation remains valid. Table S14 already correctly separates activation-error interventions from voltage-space capture; a short explicit mention of the mean-over-12 projected K1 control and the three groups of four for K3 would make the fresh comparison easy to locate.
5. Table S13's $\boldsymbol q$ row currently defines the reciprocal-cable adjoint and multi-affine path derivative, but omits the new conductance model's $q_n=\partial v_s/\partial v_n$. Add that directed-conductance definition; it is a forward path derivative before multiplication by scalar somatic error.

The conductance three-pattern row in Table S14 is otherwise accurate: six voltage coordinates, three disjoint initial-profile-weighted supports, oracle coefficients, and at most two path profiles at fixed parameters. The diagonal-gain notation distinguishes the rectangular anatomical profile from the shunt's baseline-weighted ancestry partition. No new conflict was found in those definitions.
