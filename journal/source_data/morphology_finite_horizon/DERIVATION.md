# Finite-horizon operators and their scope

This investigation leaves the original frozen experiment and manuscript unchanged. The new protocol is `protocol.json`; old development and confirmation data are diagnostic, and new seeds 9300–9319 are a separately frozen test within the same task family. The task family's expressivity and geometry limitations identified in `analysis/PREDICTIVE_MORPHOLOGY_REASSESSMENT_20260905.md` still apply.

## Reduced learner

Let the route columns be orthonormal, `Phi^T Phi=I`, and let `P=Phi Phi^T`. Starting from zero, the original update keeps `W` in the route span: `W=Phi V`. For observed context `a_j`, define `c_j=Phi^T H^T a_j`. The scalar prediction is `c_j^T V x`, and the actual projected weight update is equivalent to

`V_next = V - eta * mean_batch[c_j (c_j^T V x - y) x^T]`.

This is the correct reduced operator; the forward transfer appears inside `c_j`, and the output matrix `V` remains trainable. Omitting that compensation would instead predict frozen-field approximation error.

## Exact full-batch spectral dynamics

For an empirical fit design `z_i=vec(c_i x_i^T)`, define

`H_emp = mean(z_i z_i^T)`, `b_emp=mean(y_i z_i)`.

From zero, full-batch gradient descent has

`v_T = U diag(f_T(lambda)) U^T b_emp`,

where `H_emp=U diag(lambda) U^T`, `f_T(lambda)=[1-(1-eta lambda)^T]/lambda`, and the continuous value at zero is `eta*T`. The implementation computes the entire finite-horizon mean, including learning-rate-dependent contractions and null modes. It evaluates this vector on an explicitly separate observation set. The empirical selector fits the first 512 calibration observations and evaluates on the remaining 512. It is exact for that full-batch fit/evaluation problem, **not** for the original 2,048-observation minibatch learner.

Under the Gaussian context model, the population Hessian factors as `C ⊗ I_d`, where `C=sum_j p_j c_j c_j^T`. The smaller matrix `C` gives the corresponding exact population full-batch forecast without constructing the larger empirical Hessian.

## Exact fresh-example SGD moments

Assume each minibatch contains `B` independent new population examples. Context `j` has probability `p_j`, input `x~N(0,I_d)`, target `y=t_j^T x+epsilon`, and independent zero-mean noise with variance `sigma_j^2`. Only the noise first and second moments are required; its distribution need not be Gaussian. The input Gaussian fourth moments are essential.

Define

`M=E[V]`, `Gamma=E[(V-M)(V-M)^T]`,

`C=sum_j p_j c_j c_j^T`, `D=sum_j p_j c_j t_j^T`,

`g_bar=C M-D`, and `B_eta=I-eta C`.

For each context, the expected squared coefficient residual is

`R_j = ||M^T c_j-t_j||^2 + c_j^T Gamma c_j`.

Gaussian fourth moments give

`E_x,epsilon[(e^T x-epsilon)^2 ||x||^2] = (d+2)||e||^2+d sigma_j^2`.

Consequently the single-example gradient Gram and conditional-mean gradient Gram are

`J = sum_j p_j c_j c_j^T [(d+2) R_j+d sigma_j^2]`,

`K = C Gamma C + g_bar g_bar^T`.

The exact closed recurrence is

`M_next = M-eta g_bar`,

`Gamma_next = B_eta Gamma B_eta^T + (eta^2/B)(J-K)`.

To see the subtraction term, condition first on `V`: the Gram of a minibatch-mean gradient is `(1-1/B)` times the conditional-mean-gradient Gram plus `(1/B)` times the single-example gradient Gram. Applying total covariance yields the formula above. Dependence of the gradient noise on the current residual and weight variance is retained; this is not a constant-noise approximation.

The expected population half-MSE is

`L_T = 0.5 sum_j p_j [R_j(T)+sigma_j^2]`.

Only `K x d` means and `K x K` covariance Grams are needed. A full covariance of all `K*d` weights is unnecessary because isotropic input risk contracts it into these Grams. This closure need not hold for anisotropic inputs, nonlinear targets, correlated minibatches or a reused finite sample.

## What is estimated and what is privileged

The primary predictor fits the context probabilities, conditional coefficient vectors and noise variances from the 1,024 calibration observations. Context vectors are observed; conditional coefficients use ordinary least squares without an intercept, and noise variances use residual degrees of freedom. The known `x~N(0,I)` model is an explicit assumption. This produces an exact forecast for the fitted Gaussian population, but only a plug-in forecast for the actual generating population and finite training cache.

The population-oracle predictors use true context vectors/probabilities, teacher coefficients and label-noise variance. They diagnose the dynamics equation; they are not deployable selectors and do not estimate task complexity. Both oracle and plug-in variants are reported for full-batch and SGD dynamics.

The short-pilot selector performs 16 real updates on every candidate under each training regime and evaluates those weights on calibration observations. Its extra training work is counted. Its 16-step loss is used directly as a selection score; it is not claimed to be a calibrated endpoint-loss forecast.

All method scores add the original, unchanged resource penalty. The original scalar, development-fixed, maximum-budget and generating-rank baselines retain their original development choices. No new scalar, horizon, cost or policy was fit to old confirmation outcomes.

## Finite training cache versus fresh examples

The original learner samples minibatches with replacement from a fixed 2,048-example training cache. Even when the cache was itself sampled from a Gaussian population, later weights depend on those observations. The fresh-example recurrence is therefore not an exact unconditional formula for that procedure. The study explicitly runs both regimes. It compares predictions with both realized held-out test loss and exact population risk of the realized trained weights. Differences between the two regimes quantify this source of misspecification rather than hiding it in predictor fit.

## Validation and separation

Focused tests check: empirical spectral trajectories against direct gradient updates with a singular Hessian; population spectral predictions against full-batch recurrence; the stochastic Gram recurrence against exhaustive, exact degree-four Gaussian quadrature of all two-example minibatches from a nonzero random weight state; and a prediction input guard that rejects every request for training or test observations.

Each seed's non-pilot predictions are written and hash-sealed before its training outcomes. Fresh seeds are listed in a protocol that hashes the prediction/training code and original inputs before fresh calibration or outcome evaluation. Original-seed fixed-cache reruns are compared numerically with the retained trajectories. Analysis uses all candidate outcomes and seed-block uncertainty; it does not select successful ranks or noise conditions.
