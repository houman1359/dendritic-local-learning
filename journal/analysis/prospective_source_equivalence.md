# Prospective source-equivalence audit

Status: **verified**.

Scope: this is a dated collection-time attestation created on 5 August 2026
against the source hashes recorded below. It is not a claim that a later
moving development checkout remains byte-equivalent. The frozen manifests and
archived source copies, rather than the current development tree, define the
software used for these historical runs; the detached clean exact/BP audit
provides the current-source implementation check.

The factory change only registers unused Legendre-memory architectures. The indexed-sparse change only validates integer-valued K and chunk-size arguments that are already Python integers in every audited configuration. The spatial-morphology change only exports a helper used by recurrent architectures absent from the audited configurations. The control-plane change only registers an RTX scheduler profile that no frozen run selected. After removing those exact inactive changes, each file matches its frozen SHA-256 byte for byte.

Audited run families: 72.
Audited frozen configurations: 1900.
Architecture types: dendritic_additive, dendritic_shunting.
Scheduler profiles: kempner_dev_requeue.
Indexed-sparse configurations: 680.

| Source file | Frozen SHA-256 | Current SHA-256 | Normalized SHA-256 |
|---|---|---|---|
| `repo:src/dendritic_modeling/networks/architectures/excitation_inhibition/synapse/indexed_sparse.py` | `ca3599140e4fab2bdf947d8d777e5a213dec5afde52b3a3d53232dab54530b67` | `735ba525cffd25e1053844398103eadf96eb134b54ff92176ae90b4c14db19c6` | `ca3599140e4fab2bdf947d8d777e5a213dec5afde52b3a3d53232dab54530b67` |
| `repo:src/dendritic_modeling/networks/architectures/excitation_inhibition/synapse/spatial_morphology.py` | `94524bbc4fd814bc1ff4b4db6c97d880be240e45298d5cd4afc175a90a203d98` | `183c91ca3d96f6da6e5f6ab0bdcb12f38bf7876bf01ca16be9e19bc7560cdb46` | `94524bbc4fd814bc1ff4b4db6c97d880be240e45298d5cd4afc175a90a203d98` |
| `repo:src/dendritic_modeling/networks/architectures/factory.py` | `1345a7096d33ec72508947ff4e5c54893e15460728a3deb38b92a8e74a4a8aab` | `6b25b77d6fcb7ddd1ae7faa0880bf0a7e392ad19b90770f73db5ff746419f5f1` | `1345a7096d33ec72508947ff4e5c54893e15460728a3deb38b92a8e74a4a8aab` |
| `repo:src/dendritic_modeling/scripts/sweeps/control_plane.py` | `1231fcc8ee8f20b7fe631c3980eebb590fa2c3093d1211e85218cf1eb3261800` | `d20cd2a4af14df5fc95163f0b544ab591c555443ee4063f9a249a0ffa0d22f60` | `1231fcc8ee8f20b7fe631c3980eebb590fa2c3093d1211e85218cf1eb3261800` |
