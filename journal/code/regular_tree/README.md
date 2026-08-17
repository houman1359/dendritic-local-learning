# Regular-tree equation references

`additive_reference.py` is a standalone NumPy expression of the raw-additive
forward model, softplus chain factor, parametric-tanh derivative, and exact
child error transport. `test_additive_reference.py` validates these equations
against centered finite differences.

`source_snapshots/` contains byte-identical critical source files from the
fully archived July feedback extension:

- the shunting/raw-additive alias map;
- the forward branch dynamics;
- the learned parametric tanh;
- the 4F covariance, 5F conditional-residual, clamp, and post-factor assembly.

Each snapshot is identical both to Git commit
`c39fa57987e5f5f702274482bdd9a7a6c9e3382b` and to the corresponding file in
the working tree at package preparation. They are inspection snapshots with
repository imports, not a standalone replacement for the full training
package. The archived manifest records additional initialization and sweep
source hashes; the complete training implementation remains
`src/dendritic_modeling`.
