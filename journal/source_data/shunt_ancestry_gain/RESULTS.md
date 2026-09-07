# Focal shunting: ancestry-partition correction

The passive-tree adjoint admits an exact gain representation. For soma source
ell, baseline Green matrix R = inverse(G), and shunt site k, group each site i
by its lowest common ancestor a with k. Within that disjoint block,

    q'_i = eta_a q_i
    eta_a = 1 - kappa R_ks R_ak / [(1 + kappa R_kk) R_as].

All descendants of k share eta_k = 1/(1 + kappa R_kk). Every sister block
along the soma-to-k path shares its own gain. With h = R e_s, the dictionary
diag(h) B, where B contains the disjoint block indicators, has the diagonal
gain representation sought by the paper. Its partition indicators are
differences of nested ancestry indicators. The gain field therefore lies in
the ancestry span, but its matrix need not be diagonal in an arbitrary
overlapping route dictionary. The full synaptic gradient additionally contains
the site-dependent post-shunt driving force.

SI now gives this corollary, its proof by tree separation, the dictionary
identity and the basis/driving-force qualifications. Main focal Results and
Methods, the static-gain paragraph, focal caption, SI focal section and the
relevant glossary/claim-boundary entries use the corrected account. Other
manuscript regions were left for the parent agent's coordinated revision.

The weak-channel ensemble is now explicitly a local-linearization check at
Rm = 1,000 ohm cm² with a background leak multiplier of one. Median maximal
Na/K/Ca/HCN conductances are 0.06/0.10/0.025/0.05 times leak; NMDA is 0.15
times excitatory conductance. This is not evidence for robust operation in a
strongly regenerative regime. Every existing reported number is retained.

Four tests pass. They check 2,400 random-tree/site/dose combinations, positive
and negative or zero soma sources, equivalence of the full ancestry span,
failure of a single overlapping global column to have a scalar gain, and
within-block variation introduced by driving forces. Maximum absolute error
in the exact partition identity is 2.22e-16.

`figures/shunt_ancestry_gain_native.pdf` is the proposed main Figure 7,
518.4 by 490 pt. Panel A shows the ancestry partition; panels B–D retain the
same source data and the same original panel functions; panel E gives the
electrical boundary the full page width. The native audit reports zero
violations. Both the main and weak-channel supplementary render were visually
reviewed for labels, alignment, type and overlaps.

`figures/weak_channel_linearization_check.pdf` preserves the weak-channel and
passive overlay plus paired contrasts as a supplementary asset. The parent
should copy it to `figures/supplementary/figure_shunt_weak_channels.pdf` and
append `supplementary/shunt_weak_channel_figure.tex` after the existing SI
figure sequence, so existing numbered figures are not shifted by an
intermediate insertion. Main Figure 7's caption already describes panels A–E.

Training and all numerical source tables were untouched. New code and tests
are in `scripts/shunt_ancestry_gain/`; numerical provenance for the rebuilt
main figure is in `figures/figure_provenance.json`.
