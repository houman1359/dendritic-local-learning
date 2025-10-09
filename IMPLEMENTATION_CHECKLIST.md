# Implementation Verification Checklist

## Purpose
This checklist ensures the theoretical descriptions in `local_credit_assignment.tex` match the actual implementation in the dendritic-modeling codebase.

---

## ✅ VERIFIED: Matches Implementation

### 1. Information Factor φ (CRITICAL)
- **Paper (Eq. 11)**: $\phi_n = \frac{\Var(V_n)}{\sigma_{\text{res}}^2 + \varepsilon} = \frac{1}{1-R^2}$
- **Code (line 1261)**: `phi = var_y / residual_var`
- **Clamping**: Paper states $[0.25, 4.0]$ ✅ Code line 1262: `max(0.25, min(4.0, phi))`
- **Status**: ✅ CONSISTENT

### 2. Path Factor π
- **Paper (Eq. 26)**: Recursive $\pi_n = \pi_{n-1} \cdot R_{n-1}^{\text{tot}} \cdot \bar{g}_{n-1}^{\text{den}}$
- **Code (lines 1153-1209)**: `_compute_path_propagation_factor()`
- **Implementation detail**: Per-sample scalar (noted in paper)
- **Status**: ✅ CONSISTENT

### 3. Morphology Factor ρ
- **Paper (Eq. 8)**: $\rho_n = \frac{\Cov(\bar{V}_n, \bar{V}_0)}{\sqrt{\Var(\bar{V}_n)\Var(\bar{V}_0)} + \varepsilon}$
- **Code (lines 857-1018)**: `_compute_layer_rho()`
- **Estimators**: Batch and EMA (both documented)
- **Status**: ✅ CONSISTENT

### 4. Conductance Scaling
- **Paper (Eq. 2, 4)**: $R_n^{\text{tot}} = 1/g_n^{\text{tot}}$
- **Code**: Used in forward pass and eligibility traces
- **Status**: ✅ CONSISTENT

### 5. Driving Force
- **Paper (Eq. 6)**: $(E_j - V_n)$ for synapses, $(V_j - V_n)$ for dendrites
- **Code (lines 535-584)**: Three-factor updates use these exact forms
- **Status**: ✅ CONSISTENT

### 6. Shunting Inhibition
- **Paper (Section 1.3)**: $E_j \approx 0 \Rightarrow \partial V/\partial g_j = -xRV$
- **Code**: Same factors applied to exc/inh, sign from driving force
- **Paper explicitly notes**: "same multiplicative factors... sign difference via $(E_j - V_n)$"
- **Status**: ✅ CONSISTENT

### 7. Broadcast Error Modes
- **Paper (Section 3.1)**: 
  - (A) Scalar: $\bar{\delta} = \frac{1}{d_{\text{out}}} \sum \delta_k$
  - (B) Per-compartment: $e_n = \delta$ when $d_n = d_{\text{out}}$
  - (C) Local mismatch: $e_n = \bar{\delta} \varepsilon_n$
- **Code**: `error_broadcast_mode` config with these three options
- **Status**: ✅ CONSISTENT

### 8. Dendritic Normalization
- **Paper (Eq. 31)**: $\Delta g_j^{\text{den}} \leftarrow \frac{\Delta g_j^{\text{den}}}{\sum_k g_k^{\text{den}} + \varepsilon}$
- **Code (lines 1238-1266)**: `_compute_dendritic_normalization()`
- **Status**: ✅ CONSISTENT

### 9. Branch Type Scaling
- **Paper (Eq. 32)**: $s_j = s_{\text{basal}} + t_j(s_{\text{apical}} - s_{\text{basal}})$
- **Code (lines 1268-1290)**: `_get_branch_type_scale()`
- **Status**: ✅ CONSISTENT

### 10. Depth Modulation
- **Paper (Eq. 29)**: $\rho_j = \frac{\rho_{\text{base}}}{d_j + \alpha}$
- **Code (lines 1211-1236)**: `_compute_branch_depth_modulator()`
- **Status**: ✅ CONSISTENT

---

## 📝 SUGGESTED ADDITIONS (Not Yet in Code)

These are mentioned in the paper as alternatives or future work:

### 1. DFA-Style Random Feedback Matrix
- **Paper mentions** (Section 3.1): Optional fixed random $B_n$ for testing Theorem 5
- **Current code**: Falls back to scalar broadcast
- **Action**: Add `error_broadcast_mode = "dfa"` option
- **Priority**: Medium (for alignment experiments)

### 2. Alternative φ Formulations
- **Paper suggests** (Section 8): 
  - $\phi_n^{\text{alt}} = 1 - R^2$ (unique variance)
  - $\phi_n^{\text{cond}} = \frac{\text{HSIC}(V_n,y) - \kappa \text{HSIC}(P_n,y)}{\text{HSIC}(V_n,y)}$ (conditional HSIC)
- **Current code**: Only implements $1/(1-R^2)$
- **Action**: Add `phi_mode = "unique_variance"` or `"conditional_hsic"` options
- **Priority**: Low (research extensions)

### 3. Softplus Parameterization
- **Paper mentions** (Section 5.1): $g = \log(1 + \exp(\theta))$ alternative to $\exp(\theta)$
- **Current code**: Uses exponential only
- **Action**: Add `weight_parameterization = "softplus"` option
- **Priority**: Low (numerical stability)

---

## 🧪 EXPERIMENTAL VERIFICATION TODO

The paper proposes comprehensive experiments (Section 7). Track implementation:

### Model Suite
- [ ] Rate-based conductance (current) ✅
- [ ] Conductance-based LIF with surrogate gradients
- [ ] Multi-compartment (2-4 compartments)

### Morphology Suite  
- [ ] Chain ✅
- [ ] Balanced binary tree
- [ ] Random tree
- [ ] Reconstructed (NeuroMorpho)

### Baselines
- [ ] Backprop ✅
- [ ] Feedback Alignment
- [ ] Direct Feedback Alignment
- [ ] Target Propagation
- [ ] Equilibrium Propagation

### Metrics (to implement/automate)
- [ ] Test accuracy (tracked) ✅
- [ ] Gradient fidelity $\|\nabla_{\text{local}} - \nabla_{\text{exact}}\|_2$
- [ ] Cosine alignment per layer/epoch
- [ ] Factor dynamics ($\rho_n, \phi_n, \pi_n$ time-series)
- [ ] Morphology alignment (learned $g^{\text{den}}$ vs. anatomy)

### Ablations (to script)
- [ ] 5F baseline
- [ ] +Path
- [ ] +Depth  
- [ ] +Norm
- [ ] +Types
- [ ] Full (all enabled)
- [ ] Shunting ($E_{\text{inh}} \in \{0, -0.2, -0.4\}$)
- [ ] Broadcast modes (scalar, per-comp, DFA)
- [ ] HSIC kernels (RBF vs. linear)

### Statistical Requirements
- [ ] 5 seeds per experiment
- [ ] 95% confidence intervals
- [ ] Power analysis (detect 1-2% accuracy gap)

---

## 📊 FIGURE GENERATION TODO

The paper would benefit from these figures (not yet created):

1. **Alignment Plot**
   - X: Training epoch or layer depth
   - Y: $\cos\angle(\nabla_{\text{local}}, \nabla_{\text{exact}})$
   - Lines: Different methods (3F, 4F, 5F, +Path, etc.)

2. **Shunting Gain Curves**
   - X: Input current or voltage
   - Y: Output voltage or firing rate
   - Curves: Different $E_{\text{inh}}$ values
   - Inset: Total conductance $g_n^{\text{tot}}$

3. **Path Factor Bias**
   - Scatter: Exact path-sum vs. $\pi_n$
   - Log scale
   - Color: Layer depth
   - Density plot showing bias distribution

4. **Factor Dynamics**
   - X: Training epoch
   - Y: Factor value
   - Lines: $\rho_n, \phi_n, \pi_n$ per layer with EMA smoothing
   - Shaded: Confidence intervals

5. **Morphology Comparison**
   - Side-by-side: Chain vs. tree structures
   - Heatmap: Learned $g^{\text{den}}$ values
   - Overlay: Graph topology

---

## 🔧 CODE HYGIENE

Minor consistency improvements:

### Notation
- [ ] Review index usage ($j$ for both synapse and child in some places)
- [ ] Ensure $k$ vs $j$ vs $\ell$ is consistent
- Current: Seems acceptable, paper uses $j$ generically

### Comments
- [ ] Add references to paper equations in code comments
- Example: `# See Eq. (11) in local_credit_assignment.tex`

### Docstrings
- [ ] Update docstrings to reference paper sections
- [ ] Add "As derived in Section X" where appropriate

### Config Schema
- [ ] Ensure all paper concepts have corresponding config fields
- [ ] Add new options for suggested extensions

---

## ✅ DOCUMENTATION

- [x] Comprehensive paper draft
- [x] Implementation mapping table (Appendix A)
- [x] Example configuration (Appendix C)
- [x] This verification checklist
- [ ] Tutorial notebook walking through paper concepts
- [ ] Unit tests for each theoretical component

---

## 🎯 PRIORITY ACTIONS

**High Priority** (needed for paper submission):
1. ✅ Verify φ factor interpretation (DONE - matches code)
2. ✅ Add shunting inhibition section (DONE)
3. ✅ Extend to tree backprop (DONE)
4. ✅ Add alignment theorem (DONE)
5. Run ablation experiments (Section 7 protocol)
6. Generate key figures (4 plots minimum)

**Medium Priority** (strengthens claims):
1. Implement DFA broadcast option
2. Add gradient fidelity and alignment metrics
3. Test on reconstructed morphologies
4. Compare to FA/DFA baselines

**Low Priority** (future work):
1. Alternative φ formulations
2. Softplus parameterization
3. LIF/spiking extensions
4. Formal convergence proof

---

## 📌 NOTES

### On the φ Factor Correction
The original draft had $\phi_n = 1/(1-R^2)$ described as "unique information", which is backwards. The implementation is correct (it's an SNR-like amplification), and the paper now correctly describes it as amplifying **well-predicted** compartments. An alternative $\phi_n = 1 - R^2$ for unique-information emphasis is noted as future work.

### On Tree vs. Chain
The implementation currently uses layer-wise factors (implicit chain assumption). For true tree structures, the path-sum in Theorem 1 would need per-branch path tracking. The current $\pi_n$ uses layer-averaged conductances, which is a reasonable approximation documented in the paper.

### On Experimental Scale
The proposed experimental suite is comprehensive. Consider starting with a subset:
- 2 models (rate + LIF)
- 2 morphologies (chain + binary tree)
- 3 baselines (BP, FA, DFA)
- Core ablations (5F vs. 5F+Path+Depth+Norm)
Then expand based on initial results.

---

**Last Updated**: October 9, 2025
**Verified By**: Implementation check against local_learning.py
**Status**: Paper and code are consistent ✅
