# DendriNet selection and distractor robustness

This follow-up was motivated by the completed 540-run exploratory matrix, not
specified before those results. It retains the same production DendriNet
implementation and separates representability from restricted-credit learning.
The original matrix, including mixture failures, remains available.

## Primary experiment

Sixteen neurons each contain a `[4,2]` tree: eight terminals, four proximal
compartments and a soma. Two terminals encode the two features of each of four
streams. A one-hot cue selects a stream. The independent target is

    y = sum_b c_b (-1)^b [tanh(z_b1)+tanh(z_b2)]/2.

The original pilot's pairwise product is removed. Its identity-output,
single-feature-per-terminal configuration was conditionally additive, imposing
a population NMSE lower bound v/(8+v)=0.06081 for the pilot target, where
v=1-tanh(2)/2. This is a representation limit, not label noise or a credit
deficit. The corrected target removes that particular obstruction; finite
conductance, optimization and other approximation limits still require an
exact-BP reference.

Shunting, tonic inhibition and current injection receive identical context
information, including the excitatory cue channel. Five learning rules compare
exact BP, broadcast, relative-resistance gating, a wrong-branch gate and a
uniform within-neuron RMS gate. All conditions train conductances and readout.
Local rules detach intercompartment paths; exact local derivatives and the
supplied somatic error remain available. This is not endogenous error delivery.

The held-out stress test scales only irrelevant latent features by
s=1,1.5,2,2.5,3. Labels, cues and relevant features are identical across s.
Inputs are exp(z) and exp(-z), so latent scaling broadens the presynaptic-rate
distribution nonlinearly. It is not a claim about ordinary image-task accuracy.

## Secondary interaction experiment

The same external target gains +0.25*tanh(z_b1)*tanh(z_b2) within each stream,
and proximal outputs become tanh(Vp). Fixed parent nonlinearities permit
feature interactions without changing the active parameter count. This arm
uses shunting and all five credit rules. It is a separate task/model condition,
not an isolated nonlinearity comparison. The local gate omits the parent's
activation derivative; success or failure tests a genuine approximation beyond
the identity-parent case. A nonzero mixed sensitivity establishes capacity for
interactions, not exact representability of every target.

## Development and fresh seeds

Three new development seeds compare Adam rates 0.01,0.03,0.1 over 2,048 updates:
180 trajectories. Mean validation NMSE selects one rate per task/forward/rule.
Twenty disjoint fresh paired seeds then run the fixed settings for 4,096
updates: 400 trajectories. Training/validation/test sizes are 2,048/1,024/4,096;
batch size 128; validation every 128 updates, including initialization.
Log conductances are bounded to [-9,9]; bound contacts and norm-10 clipping are
reported. The larger bounds and budget are changes from the first pilot.

Two primary contrasts are specified before fresh outcomes: broadcast minus
local-gate NMSE and uniform-RMS minus local-gate NMSE, both at s=3 in the
separable/shunting condition. Report 95% paired seed-bootstrap intervals and
two-sided sign-flip tests with Holm correction over these two tests. The
forward factorial, wrong-branch control, exact reference, full severity curve
and interaction arm are secondary. Do not infer equivalence from similar means.
Common-state diagnostics use equal-length core updates and leave the readout
fixed; they are not used to construct or tune local training gradients.

Frozen snapshots and outputs are on kempner_project_b, with W&B disabled. The
study reuses the original immutable production snapshot. No running snapshot
is edited. `protocol.py freeze` requires all development endpoints and selects
rates using validation only. Every outcome is retained, including unfavorable
controls and representational/optimization limitations.
