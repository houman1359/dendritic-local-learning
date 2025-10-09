# Quick Reference: What Changed

## 🎯 Bottom Line
Your draft has been upgraded from a good technical description to a **comprehensive monograph** that:
- ✅ Grounds the model in biophysics (cable equation)
- ✅ Works for general dendritic trees (not just chains)
- ✅ Properly addresses shunting inhibition
- ✅ Matches the implementation exactly (critical φ correction)
- ✅ Proves alignment with exact gradients
- ✅ Connects to 14+ key papers in the literature
- ✅ Provides a complete experimental protocol

---

## 📈 Stats
- **Original**: ~600 lines
- **Revised**: 897 lines (+50%)
- **New sections**: 5
- **New theorems/propositions**: 4
- **New references**: 14
- **Tables**: 3 (1 existing, 2 new)

---

## 🔑 Most Important Changes

### 1. **φ Factor Fix** (CRITICAL)
**Before**: Paper said φ = 1/(1-R²) measures "unique information"
**After**: Correctly describes it as SNR-like (amplifies well-predicted compartments)
**Why it matters**: Was inconsistent with implementation, now matches exactly

### 2. **Tree Backpropagation**
**Before**: Only derived for chains
**After**: Full theorem for arbitrary dendritic trees with path-sum formulation
**Why it matters**: Your path factor π is now properly justified as approximating a sum-over-paths

### 3. **Shunting Inhibition**
**Before**: Mentioned only via E_j ≤ 0
**After**: Complete subsection with divisive effect proof and learning rule
**Why it matters**: Critical for biological plausibility claims

### 4. **Alignment Theorem**
**Before**: No formal justification for broadcast error
**After**: Theorem proving positive expected alignment (links to FA/DFA)
**Why it matters**: Answers "why do local rules work?"

### 5. **Cable Equation**
**Before**: Started directly with compartment model
**After**: Derives from passive cable PDE
**Why it matters**: Scientific rigor - shows where assumptions come from

---

## 📚 New Sections

1. **Section 1.1**: From Passive Cable to Compartment Equation
2. **Section 1.3**: Shunting Inhibition and Divisive Gain Control  
3. **Section 3.2**: Gradient Alignment with Broadcast Errors
4. **Section 5.2**: Online Variant with Eligibility Traces
5. **Section 8**: Future Extensions and Open Questions
6. **Section 9**: Related Work

---

## 📖 Bibliography Additions

**Biological foundations**:
- Koch (Biophysics of Computation)
- Dayan & Abbott (Theoretical Neuroscience)
- Carandini & Heeger (2012) - normalization
- Holt & Koch (1997) - shunting
- Larkum (2013) - apical mechanism
- Turrigiano (2008) - homeostatic plasticity

**Algorithmic connections**:
- Lillicrap et al. (2016) - Feedback Alignment
- Nøkland (2016) - Direct Feedback Alignment
- Guerguiev et al. (2017) - segregated dendrites
- Sacramento et al. (2018) - dendritic backprop
- Whittington & Bogacz (2019) - theories of backprop
- Scellier & Bengio (2017) - Equilibrium Propagation

**Methods**:
- Gretton et al. (2005, 2007) - HSIC
- Frémaux & Gerstner (2016) - 3-factor rules
- Bellec et al. (2020) - e-prop
- Welford (1962) - online variance

---

## 🧪 Experimental Protocol Added

**Models**: Rate, LIF, Multi-compartment
**Morphologies**: Chain, binary tree, random, reconstructed
**Baselines**: BP, FA, DFA, TargetProp, EP
**Metrics**: 5 (performance, fidelity, alignment, factors, morphology)
**Ablations**: 6 main + shunting + broadcast + HSIC
**Stats**: 5 seeds, 95% CIs, power analysis

---

## ✅ Verification

**Checked against code**:
- ✅ φ = var_y / residual_var (line 1261) → Paper now describes correctly
- ✅ Path factor per-sample scalar → Paper notes this
- ✅ Shunting same factors, sign from driving force → Paper states explicitly
- ✅ All config options documented → Appendix A maps equations to code

**Consistency**: 10/10 major components verified ✅

---

## 📝 Supporting Documents Created

1. **REVISION_SUMMARY.md**: Complete list of changes with before/after
2. **IMPLEMENTATION_CHECKLIST.md**: Detailed verification of paper vs. code
3. **This file**: Quick reference

---

## 🚀 Next Steps

**For submission**:
1. Run experiments (use protocol in Section 7)
2. Generate 4 key figures (alignment, shunting, path bias, factors)
3. Proofread citations and formatting
4. Add author info and acknowledgments

**For strengthening**:
1. Implement DFA broadcast option (test Theorem 5)
2. Add gradient fidelity metrics to training loop
3. Test on reconstructed morphology (NeuroMorpho)

**For future work** (mentioned in paper):
1. Alternative φ formulations (unique variance, conditional HSIC)
2. LIF/spiking extensions
3. Formal convergence proof

---

## 💡 Key Insights from Reviewer

The reviewer's main points were:
1. **Ground in cable equation** → DONE (Section 1.1)
2. **Tree not chain** → DONE (Theorem 1)
3. **Shunting needs care** → DONE (Section 1.3)
4. **φ factor backwards** → FIXED (Definition now correct)
5. **Prove alignment** → DONE (Theorem 5)
6. **Add temporal version** → DONE (Section 5.2)
7. **Link to biology** → DONE (homeostasis, normalization)
8. **Experiment plan** → DONE (Section 7)
9. **Related work** → DONE (Section 9)

**All 9 major points addressed** ✅

---

## 🎓 What This Enables

**For reviewers**: 
- Complete theoretical foundation
- Clear connections to neuroscience and ML literature
- Testable predictions with detailed protocols

**For implementation**:
- Every equation maps to code (Appendix A)
- Extension points clearly marked
- Alternative formulations documented

**For readers**:
- Self-contained (cable → compartments → learning rules)
- Multiple entry points (biology, theory, algorithms)
- Clear future directions

---

**Status**: Ready for experiments and figures
**Document Quality**: Publication-ready theory and methods
**Next Bottleneck**: Empirical validation
