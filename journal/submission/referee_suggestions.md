# Referee suggestions

Draft for author approval. Nothing here is a declaration. Every name is a
candidate drawn from the Article's own citation base; affiliations are those
attached to the cited work and must be re-checked on the submission date, as
must every conflict of interest. Supply names to the portal only if it asks for
them and only after all authors approve.

Article: *Dendritic morphology as a dictionary for local credit assignment*.
Authors: Houman Safaai, Maceo Richards, Bernardo L. Sabatini.

## What a referee has to be able to judge

The Article spans four bodies of expertise, and no single referee covers all of
them. A useful panel needs at least one referee from each of the first three.

1. **Dendritic credit-assignment theory.** Whether the route-dictionary
   formulation is a real advance over segregated-dendrite and
   somatic-prediction accounts, and whether the exact-path reference is the
   right control.
2. **Dendritic biophysics and inhibition.** Whether the passive-cable identity
   placing a focal shunt's adjoint effect in a baseline-weighted ancestry
   partition is correct, and whether the conductance and calibration ranges are
   physiological.
3. **Learning theory and experimental design.** Whether the paired-seed
   protocols, prespecified margins, interval definitions and the retained
   negative prospective selector support the claims made.
4. **Connectomics and functional imaging.** Whether the MICrONS reconstruction
   analyses and the measured-response null are handled correctly.

## Suggested referees

| Candidate | Affiliation at time of cited work | Area | Why this Article |
|---|---|---|---|
| Panayiota Poirazi | IMBB-FORTH | 1, 2 | Dendritic capacity and dendrites in artificial networks; cited as `poirazi2003pyramidal` and `chavlis2025dendrites`. Well placed to judge whether the dictionary formulation adds to two-layer accounts. |
| Walter Senn | University of Bern | 1, 3 | Dendritic prediction of somatic spiking and cortical microcircuit approximations to backpropagation; `urbanczik2014dendritic`, `sacramento2018dendritic`, `schiess2016somatodendritic`. The most direct theoretical comparison class. |
| Blake A. Richards | Mila and McGill University | 1, 3 | Segregated dendrites and the dendritic framing of credit assignment; `guerguiev2017segregated`, `richards2019dendritic`. |
| Idan Segev | Hebrew University of Jerusalem | 2 | Principles of synaptic inhibition in dendrites; `gidon2012inhibition`. Best placed to check the shunt-adjoint identity and the cable-state claims. |
| Michael Häusser | University College London | 2, 4 | Single-branch function and synaptic learning rules exploiting nonlinear dendritic computation; `branco2010single`, `london2005dendritic`, `bicknell2021synaptic`. |
| Richard Naud | University of Ottawa | 1, 3 | Burst-dependent coordination of learning in hierarchical circuits; `payeur2021burst`. Strong on whether a local gate can supply selection. |
| Friedemann Zenke | Friedrich Miescher Institute | 1, 3 | Disinhibitory control of the sign of plasticity; `rossbroich2023disinhibitory`. Directly relevant to the inhibitory gate. |
| Rui Ponte Costa | University of Oxford | 1 | Cell-type-specific feedback and hierarchical credit assignment; `greedy2026celltype`. |
| Bartlett W. Mel | University of Southern California | 2 | Dendritic memory capacity and the two-layer abstraction the Article tests against; `poirazi2001capacity`. |
| Casey M. Schneider-Mizell | Allen Institute for Brain Science | 4 | Connectomic census of inhibitory specificity in mouse visual cortex; `schneidermizell2025inhibitory`. See the conflict note below. |

A panel of three drawn as one from {Senn, Richards, Ponte Costa}, one from
{Segev, Häusser, Mel} and one from {Naud, Zenke, Poirazi} covers the three
required areas without duplicating a viewpoint.

## Conflicts to declare or check

These are flags, not conclusions. The authors must verify each one.

- **Same institution.** Exclude anyone at Harvard University, Harvard Medical
  School or the Kempner Institute. The candidate list above was screened
  against the affiliations recorded in the cited works only.
- **Mark T. Harnett, MIT.** `francioni2026vectorized` supplies the recorded
  data reanalyzed in Supplementary Figure S33. That makes this group both the
  most qualified reader of that analysis and an interested party. The authors
  should decide whether to suggest, to exclude, or to disclose and let the
  editor choose. Do not list this candidate without making the data
  relationship explicit.
- **MICrONS.** `microns2025`, `microns_pinky_v185`, `ding2025wiring` and
  `schneidermizell2025inhibitory` are the source of the reconstructions used
  throughout. Allen Institute and MICrONS-affiliated candidates, including
  Schneider-Mizell and Nuno Maçarico da Costa, are data providers. Suggest only
  with that relationship disclosed.
- **Predecessor submissions.** The Article discloses a predecessor preprint and
  a NeurIPS 2026 submission awaiting an official decision. Anyone who refereed
  that submission has seen an earlier version. Their identity is not knowable
  here; note the overlap to the editor rather than attempting to screen for it.
- **Recent co-authorship.** Screen every final candidate against the authors'
  co-author lists for the window the journal specifies, usually the past three
  to five years. That screen has not been performed here.

## Opposed referees

None proposed. If the authors wish to oppose a referee, Nature Communications
accepts a small number of exclusions with a brief, factual reason. Give the
reason as a competing interest or a direct competitive overlap, never as a
prediction about how someone would review.

## What is not covered

No candidate here is primarily a machine-learning methods reviewer. If the
editor wants one, the natural sources are `lillicrap2020backprop`,
`nokland2016dfa` and `scellier2017equilibrium`. The authors should decide
whether a fourth referee from that community helps or shifts the review toward
benchmark expectations the Article deliberately does not target.
