#!/usr/bin/env python3
"""Build the narrated guide for the Canvas-quality 20-minute workshop deck."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
LEGACY_BUILDER = HERE.parent / "ml_workshop_15_20min" / "build_speaker_guide.py"


NOTES = [
    {
        "time": "0:40",
        "title": "When dendritic structure helps local credit assignment",
        "purpose": "State the network-level problem and the conditional claim of the talk.",
        "say": (
            "Learning requires assigning a behavioral error to the internal variables that could have caused it. "
            "Artificial networks solve this with backpropagation. Biological learning must instead use signals "
            "available to neurons, dendritic branches, and synapses. I will ask when dendritic structure supplies "
            "useful coordinates for that local credit assignment. The answer is conditional: dendrites help when "
            "feedback has the right bandwidth, the task is aligned with the available routes, and those routes "
            "transport credit reliably."
        ),
        "read": "The four callouts preview the story: neuron coordinate, subtree address, route gain, and task alignment.",
        "question": "Are you claiming that dendrites implement backpropagation?",
        "answer": "No. Backpropagation is the exact reference that defines the information problem; the paper studies restricted, locally available alternatives.",
        "guardrail": "Present a phase-conditioned theory, not a universal dendritic advantage.",
        "transition": "First, why does a network objective create a local credit problem?",
    },
    {
        "time": "0:45",
        "title": "Learning in the brain is a credit-assignment problem",
        "purpose": "Move from one behavioral error to many possible causal weights at several biological scales.",
        "say": (
            "A circuit produces behavior through many neurons, branches, synapses, and trainable weights. The final "
            "loss says whether the behavior was wrong, but it does not say which internal variable was responsible. "
            "Learning therefore has to recover three things for each parameter: its destination, the sign of change, "
            "and the magnitude. The structural question runs from which neuron, to which branch within that neuron, "
            "and finally to which synapse. This talk focuses on that spatial credit problem rather than temporal credit."
        ),
        "read": "Follow the widening sequence from one task error to circuit, neuron, branch, and many candidate synapses.",
        "question": "Is this the same as temporal credit assignment?",
        "answer": "No. The focus here is structural credit across neurons, branches, and synapses; temporal eligibility could be added on top.",
        "guardrail": "Begin with network weights and objectives before discussing synaptic mechanisms.",
        "transition": "Backpropagation gives the exact computational solution to this problem.",
    },
    {
        "time": "0:50",
        "title": "Backpropagation solves parameter credit in artificial networks",
        "purpose": "Define the exact reference and separate local eligibility from transported error.",
        "say": (
            "For a unit with preactivation u i equal to the weighted sum of its inputs, backpropagation defines delta i "
            "as the derivative of the loss with respect to u i. The weight gradient then factorizes as delta i times "
            "the presynaptic activity x j. The x j factor is local eligibility. The difficult factor is delta i, which "
            "transports downstream consequences backward through the network using the chain rule. That eligibility "
            "times transported-error decomposition will remain intact when we move from point neurons to dendrites."
        ),
        "read": "Forward arrows compute the output; purple arrows return the exact loss sensitivity. Read the equation as eligibility times returned error.",
        "question": "Why use backpropagation if it is not a plausible neuronal mechanism?",
        "answer": "Because it supplies the exact gradient against which the direction and information content of restricted local rules can be measured.",
        "guardrail": "Do not say that backpropagation sends an independent message to every weight; eligibility makes gradients parameter-specific.",
        "transition": "Biological theories relax different requirements of that exact reverse calculation.",
    },
    {
        "time": "0:50",
        "title": "Biological theories solve different parts of credit assignment",
        "purpose": "Place the work relative to three-factor rules, feedback alignment, equilibrium methods, and dendritic-error models.",
        "say": (
            "Existing theories reduce different demands of backpropagation. Three-factor rules multiply local "
            "eligibility by a modulatory signal. Feedback alignment replaces exact symmetric weights with approximate "
            "feedback. Predictive coding and equilibrium methods infer error-related quantities from dynamics. "
            "Dendritic-error models use compartmental segregation to expose teaching signals locally. Our question is "
            "narrower and complementary: once some teaching information reaches a neuron, how many coordinates does it "
            "contain, which locations receive them, and can conductance regulate their gain within the arbor?"
        ),
        "read": "Each column changes a different part of the exact credit pathway; the dendritic destination problem remains the focus here.",
        "question": "Is this another competing learning algorithm?",
        "answer": "Primarily it is a framework for comparing the operators by which local rules distribute credit within and across neurons.",
        "guardrail": "Acknowledge the historical precedents; novelty lies in the coordinate, routing, and phase-boundary analysis.",
        "transition": "The unresolved spatial bottleneck appears when a biological neuron is represented as one point.",
    },
    {
        "time": "0:45",
        "title": "The unresolved problem begins inside the neuron",
        "purpose": "Explain the point-neuron bandwidth bottleneck without implying neuronal collapse.",
        "say": (
            "For a point neuron, every synaptic update can still combine its own local eligibility with one returned "
            "neuron-level signal. A shared signal does not make all weights or neurons identical: their inputs, "
            "voltages, initial conditions, and eligibilities remain different. The limitation is feedback bandwidth. "
            "One neuron-level coordinate cannot tell two simultaneously active branches that they need oppositely "
            "signed credit on the same example. Dendritic structure makes that missing within-neuron destination "
            "question explicit."
        ),
        "read": "The left side has many distinct eligibilities but one task-dependent coordinate; the right side exposes several unresolved destinations.",
        "question": "Would a shared scalar make the layer behave like one neuron?",
        "answer": "No. It restricts task-dependent learning to a low-dimensional direction while preserving state and eligibility diversity.",
        "guardrail": "Call this a bandwidth bottleneck, not collapse or weight equalization.",
        "transition": "A dendritic tree adds three concrete resources beyond that point-neuron description.",
    },
    {
        "time": "0:45",
        "title": "A dendritic tree adds local state, subtree addresses, and route gain",
        "purpose": "Define the biological resources before introducing any positive result.",
        "say": (
            "A dendritic neuron adds local voltage and conductance state, nested subtree addresses, and state-dependent "
            "route gain. Different compartments therefore have different eligibilities. A signal delivered to one "
            "branch can be shared by its descendants without being broadcast to the full neuron. Conductance can also "
            "change how strongly a perturbation travels along that route. But these are available resources, not "
            "guaranteed benefits. Whether they help depends on the credit pattern demanded by the task."
        ),
        "read": "The colored morphology places local state, address, gain, and task alignment directly on one arbor.",
        "question": "Should greater physical dendritic depth automatically improve learning?",
        "answer": "No. Extra depth can add attenuation and noise; it helps only when computation and credit demand match the hierarchy.",
        "guardrail": "Do not equate more compartments with more useful credit.",
        "transition": "To derive eligibility and gain, we first specify the forward conductance model.",
    },
    {
        "time": "0:55",
        "title": "A dendritic neuron is a conductance tree with explicit E/I synapses",
        "purpose": "Define excitation, inhibition, steady state, and shunting in one forward model.",
        "say": (
            "Each compartment receives nonnegative excitatory and inhibitory conductances. They differ through their "
            "reversal potentials, not by assigning a negative weight to inhibition. Current balance gives a steady-state "
            "voltage equal to reversal-weighted current divided by total conductance. Excitation and inhibition both "
            "increase the denominator and lower input resistance, while their driving forces determine current direction. "
            "Near the inhibitory reversal potential, an inhibitory synapse can displace voltage only weakly and still "
            "change gain through that denominator effect. That is the shunting mechanism studied here."
        ),
        "read": "The left tree defines compartment and synapse sets; the right equations move from current balance to the normalized steady-state voltage.",
        "question": "Is inhibition represented as a negative additive input?",
        "answer": "No. It enters as conductance times E-I minus V, so it alters both current and effective membrane conductance.",
        "guardrail": "Keep reversal potentials and the denominator effect explicit whenever excitation and inhibition are compared.",
        "transition": "Differentiating this steady state shows which part of the exact gradient is locally available.",
    },
    {
        "time": "0:50",
        "title": "Exact local eligibility has explicit excitatory and inhibitory cases",
        "purpose": "Derive the local conductance eligibility and distinguish it from transported error.",
        "say": (
            "The exact gradient for conductance g i at compartment n again factorizes. Local eligibility contains "
            "presynaptic activity x i, local input resistance R total n, and the driving force E reverse i minus V n. "
            "The remaining factor, delta V n u, is the compartment learning signal: how the loss changes if voltage at "
            "that compartment changes. Excitatory and inhibitory eligibilities can therefore have opposite local signs "
            "when E I is below V n and E E is above it. The final gradient sign still also depends on the transported "
            "compartment error."
        ),
        "read": "Read the large factorization left to right, then compare the green E and coral I driving-force terms.",
        "question": "Does a synapse need to know the whole dendritic tree to compute eligibility?",
        "answer": "No. Eligibility uses local activity, voltage, reversal potential, and input resistance; global dependence enters through the compartment error.",
        "guardrail": "Do not call eligibility alone the full gradient.",
        "transition": "The remaining question is how that compartment error is transported through the tree.",
    },
    {
        "time": "0:50",
        "title": "Tree transport is a special case of the general adjoint",
        "purpose": "Connect the intuitive path product to the exact reciprocal-compartment calculation.",
        "say": (
            "Write the steady state as F of V, g, and x equals zero. Implicit differentiation gives the adjoint system "
            "J V transpose q equals the loss gradient with respect to V. Intuitively, q n measures how much the loss "
            "would change if a small current were injected into compartment n. In a directed triangular tree, this "
            "solution factorizes into local transfer gains along the unique ancestry path. In a reciprocal conductance "
            "tree, a single path product is not general, but the full adjoint remains exact."
        ),
        "read": "The left panel gives the intuitive unique-path product; the right panel gives the general implicit-system adjoint.",
        "question": "Are the path-product and adjoint calculations different algorithms?",
        "answer": "Where both apply they are equivalent forms of the same gradient; the adjoint is the general formulation.",
        "guardrail": "Restrict exact single-path language to the directed-tree model.",
        "transition": "We can now describe restricted feedback by the field it makes available inside each neuron.",
    },
    {
        "time": "0:50",
        "title": "Restricted local learning is a routed credit operator",
        "purpose": "Put scalar, neuron-specific, subtree, and exact feedback in one notation.",
        "say": (
            "For neuron u, the available compartment field is A u times c u. The columns of A u specify route supports "
            "or addresses, c u carries their signed coefficients, and K is feedback bandwidth. One layer-wide scalar "
            "provides a single shared coordinate. Neuron-specific feedback distinguishes cells but repeats one value "
            "within each arbor. Subtree feedback provides several nested addresses. Exact compartment feedback supplies "
            "one independent value per modeled location. This notation separates the content of a learning signal from "
            "where it is delivered."
        ),
        "read": "Follow one coefficient into one colored route, then compare the four feedback resolutions along the bottom.",
        "question": "Is the route matrix A u learned?",
        "answer": "Not in the principal comparisons. It defines the prescribed route dictionary; imposed alignment is stated explicitly when used.",
        "guardrail": "Do not conflate neuron identity, arbor ownership, subtree address, and exact compartment location.",
        "transition": "The operator view lets us predict when restricting that field will help or hurt.",
    },
    {
        "time": "0:55",
        "title": "A stochastic credit operator predicts when restricted routes help",
        "purpose": "Present the signal-noise utility as the theoretical centerpiece and state its empirical validation.",
        "say": (
            "Let the exact stochastic update have mean mu and covariance Sigma, and let M be the restricted credit "
            "operator. The utility numerator is the squared alignment between useful mean gradient and routed update. "
            "The denominator penalizes routed update energy and noise, scaled by loss smoothness. The theory therefore "
            "predicts a boundary rather than a fixed ranking: extra routes help only when they capture aligned signal "
            "faster than noise. Across 540 controlled conditions, utility ranks one-step progress with Spearman 0.937 "
            "and final outcomes with 0.916."
        ),
        "read": "The left schematic routes signal and noise through M; the scatter tests the resulting utility against observed progress.",
        "question": "Does this bound predict exact final accuracy?",
        "answer": "No. It is exact for a quadratic loss and most directly predicts one-step geometry; the final-outcome correlation is empirical validation.",
        "guardrail": "Call this a predictive bound and boundary map, not a fitted law of all training trajectories.",
        "transition": "Ordinary benchmark tasks first reveal which feedback coordinate is actually missing.",
    },
    {
        "time": "0:45",
        "title": "Neuron identity dominates standard tasks",
        "purpose": "Show that standard image tasks demand neuron identity but little within-arbor resolution.",
        "say": (
            "Across MNIST and flattened CIFAR-10, replacing a layer-wide scalar with neuron-specific feedback improves "
            "accuracy by roughly eight to sixteen percentage points. Once neuron identity is available, replacing the "
            "neuron-level signal with the exact compartment field changes accuracy by less than one point. These tasks "
            "therefore require the learning system to distinguish neurons, but they provide little evidence that it must "
            "distinguish exact locations within each arbor. This null motivates a task that creates opposing branch-level "
            "credit inside one neuron."
        ),
        "read": "Compare adjacent rungs within each dataset: scalar to neuron-specific, then neuron-specific to exact field.",
        "question": "Does this show that dendrites are unnecessary?",
        "answer": "Only for the within-arbor credit demands of these benchmark tasks and architectures; it does not generalize to every task or dendritic computation.",
        "guardrail": "Treat the exact-path null as a result, not as something to hide.",
        "transition": "To demand an internal address, we make simultaneously active branches require conflicting updates.",
    },
    {
        "time": "0:50",
        "title": "Credit conflict creates a demand for branch-specific signals",
        "purpose": "Define the controlled branch-conflict task and derive its shared-mode crossover.",
        "say": (
            "All B branches are active. Context c selects the branch that determines the label, while chi controls how "
            "often nonselected branches carry opposite-class evidence. At chi zero, the branch updates are compatible. "
            "At chi one, sibling branches demand opposing credit. Summing those directions gives a shared-mode coefficient "
            "B minus two chi times B minus one, which changes sign at chi c equal to B over two times B minus one. Above "
            "that crossover, one neuron-shared signal becomes noninformative while a branch-specific route can preserve "
            "the correct local sign."
        ),
        "read": "Compare the compatible and conflicting examples on the left, then read the two equations as the shared-credit cancellation condition.",
        "question": "Was this task constructed to favor branch-specific credit?",
        "answer": "Yes, deliberately. It is a mechanism-matched existence test of the condition the theory says should create path demand.",
        "guardrail": "Present this as a controlled stress test, not a claim that every natural task has this structure.",
        "transition": "The trained networks now let us test whether failure occurs at that predicted boundary.",
    },
    {
        "time": "0:45",
        "title": "Shared credit fails at the predicted conflict boundary",
        "purpose": "Show agreement between the analytic crossover, trained collapse, and ownership control.",
        "say": (
            "The shared rule collapses near the analytic threshold for two, four, and eight branches. At full conflict, "
            "the correct branch route improves performance by thirty-five to fifty-eight percentage points. A cyclic "
            "derangement preserves rank and sparsity but delivers each signal to the wrong branch, and it fails. This "
            "shows that additional bandwidth alone is not enough: each signal must reach the branch whose local "
            "eligibility it is intended to modulate. The main result is the alignment between the predicted sign change "
            "and the trained transition."
        ),
        "read": "The left plot shows learning collapse at the dashed thresholds; the right plot compares analytic and trained crossings.",
        "question": "Could the gain come simply from adding feedback parameters?",
        "answer": "No. The cyclic derangement preserves bandwidth and sparsity while breaking ownership, so correct delivery is causal.",
        "guardrail": "The result proves a need for an address, not a unique material need for a dendritic tree.",
        "transition": "A hierarchical task asks when nested ancestry is an efficient address basis rather than merely a correct one.",
    },
    {
        "time": "0:50",
        "title": "The right subtree address matters; fine topology adds a narrow gain",
        "purpose": "Separate the large ownership effect from the small topology-specific advantage.",
        "say": (
            "The hierarchical task organizes eight streams and exposes K independent route coefficients. At K equals "
            "four, correct ancestry assignment beats derangement by 61.1 percentage points, so ownership can be essential. "
            "But against the strongest rank- and sparsity-matched non-anatomical basis, ancestry wins by only 1.27 points. "
            "It loses at low bandwidth, wins narrowly at the intermediate budget, and ties at full rank. The robust result "
            "is therefore hierarchy-resolution matching: correct addresses carry most of the benefit, while fine tree "
            "topology adds a small conditional increment."
        ),
        "read": "Read the task hierarchy on the left, then compare the green correct, purple matched, and gray deranged curves across K.",
        "question": "Why is the derangement effect so much larger than the matched low-rank effect?",
        "answer": "Derangement destroys who owns a signal; the matched low-rank basis keeps comparable bandwidth and approximates much of the useful subspace.",
        "guardrail": "Always pair the 61.1-point ownership contrast with the 1.27-point topology-specific contrast.",
        "transition": "If anatomy is to support routes, reconstructed arbors should at least provide an efficient candidate dictionary.",
    },
    {
        "time": "0:50",
        "title": "Real arbors provide sparse candidate routes",
        "purpose": "Present route capacity and wiring economy without claiming endogenous biological use.",
        "say": (
            "We extract ancestry-defined routes from reconstructed arbors and ask how much modeled credit-field energy "
            "their span captures. The route dictionary retains about eighty-five percent of dense capture using roughly "
            "seven percent of the dense route-matrix connections. The model-matched capture-per-wire ceiling is 14.2-fold; "
            "the stricter density-matched anatomical comparison is about 2.7-fold. The main analysis uses a disjoint "
            "47-cell cohort, and the ordering recurs descriptively in ten cells from a second mouse. This establishes "
            "sparse route capacity, not observed use of those routes during learning."
        ),
        "read": "The morphology shows candidate route supports; the center defines capture and reports both model-matched and density-matched efficiencies.",
        "question": "Does high capture show that real neurons use these routes for credit assignment?",
        "answer": "No. It shows representational capacity of modeled fields on measured anatomy, not endogenous task gradients or plasticity.",
        "guardrail": "Pair 14.2-fold with the approximately 2.7-fold density-matched control and call the second-mouse result descriptive.",
        "transition": "Anatomy can supply addresses; conductance may regulate the gain along an already available route.",
    },
    {
        "time": "0:50",
        "title": "Focal shunting changes descendant credit only in permissive regimes",
        "purpose": "Explain the matched intervention and foreground the standard-passive null.",
        "say": (
            "We compare a focal inhibitory conductance with an additive current matched to its baseline focal current, "
            "then restore somatic voltage after both interventions. Only the shunt changes the conductance matrix and "
            "therefore the adjoint field. In the directed tree, the first-order log-gain change is restricted to "
            "descendants whose route crosses the inhibited site. Empirically, the standard passive calibration at R m "
            "equal to fifteen thousand is null. Descendant localization appears only at lower resistance or with "
            "background conductance. Shunting is therefore a conditional gain mechanism, not a generic learning advantage."
        ),
        "read": "Compare the two vector schematics, read the route-local derivative, then locate the standard-passive point near zero on the boundary plot.",
        "question": "Is the main focal-shunting result positive or null?",
        "answer": "At the standard passive calibration it is null; the positive localization appears only in a defined permissive conductance regime.",
        "guardrail": "Never imply that ordinary passive parameters guarantee selective shunting or improved training.",
        "transition": "Capacity and conditional gain still do not show that measured cortical responses are aligned with those routes.",
    },
    {
        "time": "0:50",
        "title": "Measured visual responses show no morphology-specific alignment",
        "purpose": "State the biological null clearly and define its limited inferential scope.",
        "say": (
            "We map measured presynaptic visual responses onto seven target-cell reconstructions from one MICrONS mouse "
            "and predict held-out postsynaptic responses. Nested subtree routes do not outperform random or site-shuffled "
            "matched controls; the topology effects cross zero. This is a useful boundary result. Morphology supplies "
            "possible addresses, but these measured responses do not reveal preferential alignment between those "
            "addresses and the modeled functional relation. The cohort is small and does not measure plasticity, so the "
            "result is a scoped null rather than a broad rejection of dendritic learning."
        ),
        "read": "Follow measured inputs into the reconstruction, then compare held-out prediction error for exact, nested, and matched control routes.",
        "question": "Does this rule out morphology-specific credit assignment in cortex?",
        "answer": "No. It finds no detectable advantage in this seven-cell, one-mouse assay; other tasks, states, cells, or direct plasticity measurements could differ.",
        "guardrail": "Say seven cells and one mouse aloud; do not generalize beyond this cohort and assay.",
        "transition": "We therefore test the narrower causal hypothesis that alignment itself is the missing variable.",
    },
    {
        "time": "0:45",
        "title": "Imposed alignment makes the same anatomical dictionary useful",
        "purpose": "Demonstrate representational sufficiency without implying learned or endogenous alignment.",
        "say": (
            "We rotate a fixed-energy task-credit field from a direction orthogonal to the anatomical route span into a "
            "direction inside it. Anatomy, route count, field energy, and curvature remain fixed; only imposed alignment "
            "a changes. Capture and predicted progress then rise together, with median Spearman 0.99, and all eight cases "
            "lose at zero alignment but win at full alignment. This rescue explains the measured-response null and shows "
            "that the route dictionary is representationally sufficient when the geometry is matched. It does not show "
            "that cortical learning creates this alignment."
        ),
        "read": "The left trees rotate the signed field into the route span; the right plot shows capture and one-step progress coinciding.",
        "question": "Is this a trained biological-alignment result?",
        "answer": "No. It is a controlled representational intervention, not an endogenous or longitudinal learning measurement.",
        "guardrail": "Use the phrase imposed-alignment rescue or conditional representational sufficiency.",
        "transition": "The benchmark nulls, controlled positives, and biological boundary now fit on one common map.",
    },
    {
        "time": "0:55",
        "title": "Dendritic benefit lies in a matched alignment-bandwidth regime",
        "purpose": "Synthesize the results into one final answer rather than a list of effects.",
        "say": (
            "The horizontal axis is task-route alignment; the vertical axis is feedback bandwidth relative to effective "
            "task-credit rank. At low bandwidth, credit lacks the coordinates needed to distinguish neurons or branches. "
            "At weak alignment, anatomical routes capture little useful signal. At full rank, generic operators already "
            "span the field, so anatomy loses a unique advantage. Dendritic structure is most useful in the intermediate "
            "matched regime: task-relevant credit aligns with available subtree routes, bandwidth is restricted but "
            "sufficient, and transport remains reliable. The regime background is theoretical, not fitted to the points. "
            "The main contribution is this predictive boundary map, not a claim that dendrites replace backpropagation."
        ),
        "read": "Move right for greater task-route alignment and upward for greater returned bandwidth; locate each positive and null result in the corresponding regime.",
        "question": "What is the strongest experimental prediction from this map?",
        "answer": "Branch-specific plasticity should grow with task-route alignment at intermediate resolution, while focal inhibition should modulate it mainly in permissive high-conductance states.",
        "guardrail": "The phase plane is a theoretical synthesis, not a fitted claim that natural learning occupies the favorable regime.",
        "transition": "End here: coordinate, address, and gain matter only when the task and route geometry match.",
    },
]


def main() -> None:
    spec = importlib.util.spec_from_file_location("legacy_speaker_builder", LEGACY_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {LEGACY_BUILDER}")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    builder.HERE = HERE
    builder.PNG_DIR = HERE / "png"
    builder.MARKDOWN = HERE / "speaker_notes.md"
    builder.HTML_OUT = HERE / "speaker_guide.html"
    builder.PDF_OUT = HERE / "dendritic_credit_canvas_rebuild_speaker_guide.pdf"
    builder.NOTES = NOTES
    builder.STYLE += """
.body { grid-template-columns: 36% 64%; gap: .26in; margin-top: .22in; }
.thumb { border-radius: 10px; }
.block { margin-top: .16in; padding: .13in .15in; }
.label { font-size: 10.5px; margin-bottom: 5px; }
p { font-size: 13px; line-height: 1.38; }
.script { padding: .18in .20in; }
.script p { font-size: 15px; line-height: 1.42; }
.qa { gap: .15in; margin-top: .17in; }
.transition { margin-top: .17in; padding: .13in .16in; }
"""

    original_markdown = builder.build_markdown
    original_html = builder.build_html

    def build_markdown() -> None:
        original_markdown()
        text = builder.MARKDOWN.read_text(encoding="utf-8")
        text = text.replace(
            "Scripted time: 17:55. The remaining 2:05 is reserved for pauses, emphasis, and slide transitions.",
            "Scripted time: 16:10. The remaining 3:50 is reserved for equations, emphasis, and transitions.",
        )
        builder.MARKDOWN.write_text(text, encoding="utf-8")

    def build_html() -> None:
        original_html()
        text = builder.HTML_OUT.read_text(encoding="utf-8")
        text = text.replace("Kempner Learning Dynamics Workshop", "Dendrites and Credit Assignment Workshop")
        builder.HTML_OUT.write_text(text, encoding="utf-8")

    builder.build_markdown = build_markdown
    builder.build_html = build_html
    builder.main()


if __name__ == "__main__":
    main()
