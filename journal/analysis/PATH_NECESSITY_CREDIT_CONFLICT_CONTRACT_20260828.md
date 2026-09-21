# Path-necessity credit-conflict experiment contract

Frozen: 28 August 2026, after a three-seed design/optimizer screen and before
the canary or any confirmatory-seed outcome was inspected.

## Question

When does one teaching coordinate per neuron cease to be sufficient? The
experiment is designed to connect the paper's ordinary-task nulls to its
constructed path-routing successes with a continuous, analytically predicted
boundary.

Each example presents a nonzero branch-local Fashion-MNIST view to each of two,
four or eight modeled branches. A balanced context gate selects one branch, and the selected image's
binary class (T-shirt/top versus shirt) is the target. On a nonselected branch,
the input is either a copy of the selected image or an independent image of the
opposite class. The probability of the latter is the credit-conflict dose
\(\alpha\). Context and class are exactly balanced and independent.

The forward logit is

\[
z_i = w_{c_i}^{\mathsf T}x_{i,c_i},
\]

where \(c_i\in\{1,\ldots,B\}\) and \(B\in\{2,4,8\}\). For binary cross-entropy,
\(\delta_i=\sigma(z_i)-y_i\), exact path transport gives

\[
\nabla_{w_b}L = \frac{1}{N}\sum_i
\delta_i\,\mathbf 1[c_i=b]x_{i,b}.
\]

The gain-normalized neuron-shared rule replaces the one-hot path coefficient
by \(1/B\):

\[
\widetilde{\nabla}_{w_b}L = \frac{1}{N}\sum_i
\delta_i\,\frac{1}{B}x_{i,b}.
\]

This is the orthogonal rank-one projection of the exact path coordinate. It is
also algebraically equivalent to repeating the unscaled somatic error on all
branches and dividing its learning rate by \(B\); the normalization therefore
does not create a distinct information channel.

For centered class means \(\pm\mu\), the expected branch compatibility matrix
has diagonal entries 1 and off-diagonal entries \(1-2\alpha\):

\[
C_\alpha=(1-2\alpha)\mathbf 1\mathbf 1^{\mathsf T}+2\alpha I.
\]

Its shared-mode eigenvalue is
\(\lambda_{\rm shared}=B-2\alpha(B-1)\), whereas each of the \(B-1\)
contrast modes has eigenvalue \(2\alpha\). Thus \(\alpha=0\) is a rank-one
compatible regime, branch-contrast demand grows with \(\alpha\), and the
shared mode changes sign at the prespecified boundary
\(\alpha_c=B/[2(B-1)]\). The predicted boundaries are 1, 2/3 and 4/7 for
\(B=2,4,8\), respectively. Importantly, \(\alpha\) changes
eligibility--credit compatibility, not the algebraic rank of the per-example
exact path field, which remains one-hot.

## Frozen design

- Two, four or eight branches; conflict doses
  \(\alpha\in\{0,.25,.50,4/7,.60,2/3,.75,1\}\).
- Fashion-MNIST classes 0 and 6, average-pooled from 28 by 28 to 7 by 7 and
  standardized using the training images only.
- 8,192 generated training trials and 2,000 generated test trials per seed;
  selected and conflicting views are sampled with replacement from the
  official Fashion-MNIST train and test splits, respectively.
- Conflict masks are generated once as uniforms and thresholded, so every
  higher dose contains all conflicts at every lower dose.
- Selected images, initialization and data streams are paired across doses and
  learning rules.
- Full-batch gradient descent, 250 updates, learning rate 0.04, no outcome-based
  early stopping.
- One artifact-only canary seed (10999) and twenty fresh paired inferential
  seeds (11000--11019).
- No Weights & Biases logging. Raw outputs and logs reside on the
  `kempner_project_b` filesystem.

The five frozen conditions are a neuron-shared rank-one coordinate, the
correct path-resolved coordinate, a cyclic within-neuron derangement with the same
rank and sparsity, analytic backpropagation, and an explicitly gated grouped
point implementation. Backpropagation and grouped point use the same one-hot
route as correct path transport; they are implementation-equivalence checks,
not independent evidence.

## Outcomes and decisions

The primary endpoint is held-out accuracy. The primary estimand is the
seed-wise slope of correct-path minus neuron-shared accuracy on \(\alpha\).
At each dose we also report paired accuracy and loss effects, initial gradient
cosine, squared cosine (capture), and signed utility relative to the exact
gradient.

The result supports conditional path necessity only if at least 16/20
seed-wise interaction slopes are positive at each branch count, correct path
exceeds shared credit by at least 10 percentage points at \(\alpha=1\),
correct path exceeds the rank-matched derangement at \(\alpha=1\), and all
exact/BP/grouped-point equivalence checks pass. A difference within 2
percentage points at \(\alpha=0\) is the prespecified practical-equivalence
criterion for the compatible regime. The stronger phase-theory prediction is
that the trained shared-credit transition shifts left as branch count rises,
in the order \(B=8\), \(B=4\), \(B=2\). All doses are reported regardless of
outcome.

Failure of the interaction or the high-conflict contrast is a negative result.
Failure of zero-conflict equivalence means that path routing helps even before
the designed compatibility boundary and weakens the clean phase-transition
interpretation. A shifted empirical crossover does not by itself fail the
experiment: \(B/[2(B-1)]\) is the centered-class analytic prediction, whereas
image covariance and finite optimization can shift each trained boundary.

## Scope

This is a controlled sufficiency and boundary test. A positive result would
show that path-resolved credit becomes necessary when different simultaneously
driven branches require incompatible teaching coefficients. Equality of the
gated point control would show, by design, that dendritic material is not the
only way to provide this resource. The experiment does not estimate how often
natural tasks or biological neurons occupy the high-conflict regime.

## Post-run wording clarification

Here “simultaneously driven” means that every branch receives a nonzero view.
Only the context-selected branch contributes to the somatic logit. This
clarification changes neither the frozen design nor the executed code. It also
makes the scope explicit: the experiment requires branch-selective information
somewhere in the local update, but does not distinguish a routed teaching
coefficient from an equivalent branch-local eligibility gate.

The frozen contract and configuration call the conflict dose \(\alpha\). The
manuscript denotes the same quantity by \(\chi\), reserving \(\alpha\) for its
separate physical-depth sensor-alignment parameter.
