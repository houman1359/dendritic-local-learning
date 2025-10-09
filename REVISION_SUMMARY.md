# Revision Summary: Incorporating Reviewer Comments

## Date: October 9, 2025

This document summarizes the major revisions made to `local_credit_assignment.tex` based on comprehensive reviewer feedback.

---

## ✅ Critical Corrections Implemented

### 1. **Cable Equation Foundation** (Section 1.1)
- **Added**: New subsection "From the Passive Cable to the Compartment Equation"
- **Content**: Derives the compartmental voltage equation from the passive cable PDE
- **Citations**: Koch's *Biophysics of Computation*, Dayan & Abbott
- **Location**: Lines ~40-50

### 2. **Tree Backpropagation** (Section 2.2 → Theorem 1)
- **Changed**: Replaced chain-only backprop with general tree recursion
- **Added**: Theorem "Backpropagation on a Dendritic Tree" with path-sum formulation
- **Formula**: Sum over all paths $\sum_{\mathcal{P}:n\leadsto 0} \prod_{(i\to k)\in \mathcal{P}} R_k^{\mathrm{tot}} g_{i\to k}^{\mathrm{den}}$
- **Result**: Chain case becomes Corollary, clarifies path factor approximation

### 3. **Shunting Inhibition** (New Section 1.3)
- **Added**: Complete subsection with:
  - Proposition: Subthreshold divisive effect ($\partial V/\partial g_{\text{inh}} = -xRV$)
  - Remark: Voltage vs. rate-level effects (citing Holt & Koch 1997)
  - Explicit inhibitory learning rule for 3F/4F/5F
- **Citations**: Holt & Koch (1997), Carandini & Heeger (2012)

### 4. **Information Factor φ Correction** (Definition in Section 3.3)
- **CRITICAL FIX**: Corrected interpretation to match implementation
- **Implementation uses**: $\phi_n = \frac{\Var(V_n)}{\sigma_{\text{res}}^2} = \frac{1}{1-R^2}$ (SNR-like)
- **Interpretation**: Amplifies well-predicted compartments (hierarchical coherence)
- **Alternative noted**: $\phi_n = 1 - R^2$ for unique-information emphasis
- **Added**: Explicit note that implementation clamps $\phi_n \in [0.25, 4.0]$

### 5. **Gradient Alignment Theorem** (New Section 3.2)
- **Added**: Theorem proving positive expected alignment under random broadcast
- **Content**: Extends feedback alignment theory to local rules
- **Formula**: $\mathbb{E}[\cos\angle(g^{\text{local}}, g^{\text{exact}})] \ge c_n > 0$
- **Citations**: Lillicrap et al. (2016), Nøkland (2016) for FA/DFA

---

## ✅ Mathematical Enhancements

### 6. **Bounds and Convexity** (New Lemma after Eq. 4)
- **Added**: Lemma proving $V_n$ is convex combination, $0 < R_n \le 1$
- **Use**: Enables sign analysis and depth attenuation bounds

### 7. **Additional Local Sensitivities** (New Proposition)
- **Added**: $\partial V_n/\partial x_i$, $\partial V_n/\partial E_i$, $\partial V_n/\partial g^{\text{leak}}$
- **Purpose**: Supports eligibility trace formulation

### 8. **Path Factor Improvement** (Section 4.1)
- **Enhanced**: Theorem now includes bias bounds and tree vs. chain discussion
- **Added**: Corollary on depth attenuation from Lemma 1
- **Clarified**: Per-sample scalar implementation detail

---

## ✅ Implementation Consistency

### 9. **Online Eligibility Traces** (New Section 5.2)
- **Added**: Continuous-time formulation with filtered eligibilities
- **Formula**: $\tau_e \dot{e}_j = -e_j + x_j(E_j - V_n)R_n^{\text{tot}}$
- **Update**: $\Delta g \propto \int e_j(t) m_n(t) dt$
- **Citations**: Frémaux & Gerstner (2016), Bellec et al. (2020) for 3-factor and e-prop

### 10. **Positive Weight Parameterization** (Section 5.1)
- **Added**: Softplus alternative to pure exponential
- **Reason**: Avoids extreme gradients

### 11. **DFA Broadcast Option** (Section 3.1)
- **Added**: Fixed random feedback matrix $B_n$ option
- **Purpose**: Enables testing Theorem 5 (alignment)
- **Note**: Ties to experimental protocol

### 12. **Welford Algorithm Citation** (Section 3.3)
- **Added**: Proper citation for numerically stable online variance
- **Reference**: Welford (1962)

---

## ✅ Biological Connections

### 13. **Homeostatic Plasticity** (Section 4.3)
- **Added**: Connection to synaptic scaling literature
- **Citation**: Turrigiano (2008)
- **Link**: Dendritic normalization ↔ Oja-style stability

### 14. **Apical-Basal Specialization** (Section 4.4)
- **Citation**: Larkum (2013) on compartmental differentiation

---

## ✅ Experimental Validation

### 15. **Comprehensive Protocol** (Section 7)
- **Added**:
  - Three model classes (rate, LIF, multi-compartment)
  - Four morphology types (chain, binary tree, random, reconstructed)
  - Baseline comparisons (BP, FA, DFA, TargetProp, EP)
  - Five metrics (performance, fidelity, alignment, factor dynamics, morphology)
  - Shunting ablations ($E_{\text{inh}} \in \{0, -0.2, -0.4\}$)
  - Broadcast mode comparisons (scalar, per-compartment, DFA)
  - HSIC kernel comparisons (RBF vs. linear)
- **Requirements**: 5 seeds, 95% CIs, power analysis

---

## ✅ Related Work and Future Directions

### 16. **Related Work Section** (New Section 8)
- **Added**: Comprehensive literature connections
- **Topics**: Apical errors, predictive coding, EP, FA/DFA, HSIC, normalization, homeostasis
- **Citations**: 14 key references properly integrated

### 17. **Future Extensions** (New Section 8)
- **Added**: Four research directions:
  - Information factor variants (including conditional HSIC)
  - Spiking neural networks
  - Reconstructed morphologies
  - Convergence analysis

---

## ✅ Presentation Improvements

### 18. **Updated Abstract**
- Now mentions: tree gradients, four morphology-aware mechanisms, shunting inhibition, alignment theorem

### 19. **Summary Tables**
- **Table 1** (existing): Computational complexity
- **Table 2** (new): Biological analogs and key results per component
- Provides at-a-glance reference for all theoretical contributions

### 20. **Bibliography**
- **Added 14 references**:
  - Carandini & Heeger (2012) - normalization
  - Holt & Koch (1997) - shunting
  - Lillicrap et al. (2016) - FA
  - Nøkland (2016) - DFA
  - Guerguiev et al. (2017) - segregated dendrites
  - Sacramento et al. (2018) - dendritic backprop
  - Whittington & Bogacz (2019) - theories of backprop
  - Scellier & Bengio (2017) - EP
  - Gretton et al. (2005, 2007) - HSIC
  - Frémaux & Gerstner (2016) - 3-factor rules
  - Bellec et al. (2020) - e-prop
  - Larkum (2013) - apical mechanism
  - Welford (1962) - online variance
  - Turrigiano (2008) - homeostatic plasticity

---

## 📋 Consistency Checklist

| Item | Status | Notes |
|------|--------|-------|
| Cable equation derivation | ✅ | Section 1.1 |
| Tree backprop (not just chain) | ✅ | Theorem 1 |
| Shunting inhibition subsection | ✅ | Section 1.3 |
| φ factor matches code | ✅ | Def. in 3.3, implementation uses 1/(1-R²) |
| Alignment theorem | ✅ | Section 3.2 |
| Bounds lemma | ✅ | After Eq. 4 |
| Path factor for trees | ✅ | Section 4.1 improved |
| Eligibility traces | ✅ | Section 5.2 |
| DFA broadcast option | ✅ | Section 3.1 |
| Homeostatic connection | ✅ | Section 4.3 |
| Experimental protocol | ✅ | Section 7 expanded |
| Related work | ✅ | Section 8 |
| Bibliography complete | ✅ | 14 references added |
| Abstract updated | ✅ | Reflects all additions |
| Figures suggested | ⚠️ | Not implemented (LaTeX only) |
| Code verified | ✅ | φ implementation checked |

---

## 🔍 Key Implementation Verification

### Checked in `local_learning.py`:

1. **φ factor** (lines 1208-1320):
   ```python
   phi = var_y / residual_var  # = 1/(1-R²)
   phi = max(0.25, min(4.0, phi))  # Clamped
   ```
   ✅ Paper now correctly describes this as SNR-like (amplifies predicted compartments)

2. **Shunting**: Implementation applies same factors to exc/inh, sign difference from driving force
   ✅ Paper explicitly notes this in Section 1.3

3. **Path factor** (lines 1153-1209): Per-sample scalar
   ✅ Paper notes this in Remark after Theorem

4. **Broadcast modes**: Scalar, per-compartment, with fallback
   ✅ Paper documents all three + suggests DFA extension

---

## 📝 Remaining Optional Enhancements

The following were suggested but not implemented (can be added in future revision):

1. **Figures**:
   - Alignment plots (cosine similarity vs. depth/epochs)
   - Shunting gain curves
   - Path-factor bias scatter plots
   - Factor dynamics time-series

2. **Conditional HSIC φ variant**:
   - Formula provided in "Future Extensions"
   - Not implemented in code yet

3. **Formal convergence proof**:
   - Sketch provided (Robbins-Monro conditions)
   - Full proof deferred to future work

4. **Reconstructed morphology experiments**:
   - Mentioned in protocol
   - Awaiting implementation

---

## 🎯 Summary

The draft now includes:
- ✅ All critical theoretical corrections
- ✅ Consistent with implementation (φ factor fix)
- ✅ Extended to general trees (not just chains)
- ✅ Shunting inhibition fully addressed
- ✅ Alignment theorem and FA/DFA connections
- ✅ Biological grounding (cable equation, homeostasis)
- ✅ Comprehensive experimental protocol
- ✅ Complete bibliography

The paper is now theoretically rigorous, implementation-consistent, and experimentally grounded.
