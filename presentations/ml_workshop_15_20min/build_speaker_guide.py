#!/usr/bin/env python3
"""Build the slide-by-slide workshop speaker guide.

The guide is deliberately separate from the projected deck.  Each landscape
page contains the corresponding slide thumbnail, a timed script, the visual
reading, one likely audience question and answer, a claim guardrail, and the
transition to the next slide.
"""

from __future__ import annotations

import html
import shutil
import subprocess
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
PNG_DIR = HERE / "png"
MARKDOWN = HERE / "speaker_notes.md"
HTML_OUT = HERE / "speaker_guide.html"
PDF_OUT = HERE / "dendritic_credit_ml_workshop_speaker_guide.pdf"


NOTES = [
    {
        "time": "0:20",
        "title": "When can dendritic structure help local credit assignment?",
        "purpose": "State the question and the conditional answer before introducing details.",
        "say": (
            "I will ask when dendritic structure can help a network learn from locally "
            "available signals. We will move from backpropagation to local learning and then "
            "to dendritic address and gain. The answer is conditional: useful routes require "
            "both sufficient feedback bandwidth and task-route alignment."
        ),
        "read": "The subtitle previews the argument: coordinate, address, route gain, then the alignment boundary.",
        "question": "Do dendrites replace backpropagation?",
        "answer": "No. Backpropagation is our exact reference. We test when restricted, dendrite-compatible feedback retains enough of that information to support learning.",
        "guardrail": "Do not claim a biological implementation of backpropagation.",
        "transition": "First, what information problem does any learning rule have to solve?",
    },
    {
        "time": "0:50",
        "title": "Learning changes network weights; credit assigns each change",
        "purpose": "Define credit assignment in language shared by machine learning and computational neuroscience.",
        "say": (
            "A neural network maps an input to an output through many trainable weights. "
            "Training begins with one global loss, but the optimizer needs a separate update "
            "for every weight. Credit assignment is this conversion: for each parameter, "
            "which parameter should change, in which direction, and by how much? The equation "
            "at the bottom is ordinary gradient descent. The difficult object is not the "
            "forward activity itself; it is the derivative of the global loss with respect "
            "to a parameter that may be many stages upstream. This is the common problem that "
            "backpropagation and proposed neuronal learning rules must address."
        ),
        "read": "Follow the forward arrows to the loss, then the highlighted reverse route to one weight. Read the equation as delta w i equals minus eta times the loss derivative.",
        "question": "Is credit assignment just another name for gradient descent?",
        "answer": "Gradient descent specifies how to use a derivative. Credit assignment is the problem of computing or approximating the parameter-specific information that derivative requires.",
        "guardrail": "Keep the distinction between the optimization rule and the mechanism that supplies its derivative.",
        "transition": "Backpropagation tells us exactly what that returned information would be.",
    },
    {
        "time": "1:00",
        "title": "Backpropagation defines the exact neuron-specific learning signal",
        "purpose": "Introduce the exact reference and situate the work among alternative credit-assignment approaches.",
        "say": (
            "For a point neuron u, the downstream network returns delta u, the derivative of "
            "the loss with respect to that neuron's output. The reverse recursion multiplies "
            "downstream errors by synaptic weights and activation derivatives, so delta u is "
            "specific to the neuron and generally nonlocal. We use this as an information "
            "reference, not as a claim that a brain literally runs backpropagation. Existing "
            "approaches replace or infer this signal in different ways: fixed feedback, "
            "state-based objectives, eligibility traces with modulatory factors, or dendritic "
            "teaching signals. Our question is complementary: once feedback is restricted, "
            "what spatial coordinates can a neuron and its dendrites provide?"
        ),
        "read": "The recursion sums over neurons v downstream of u. Delta u belongs to a neuron, not yet to an individual incoming weight.",
        "question": "Why is delta u not already weight-specific?",
        "answer": "All incoming weights of neuron u share the same downstream loss derivative. Their different presynaptic activities and local derivatives make their final gradients distinct.",
        "guardrail": "Do not imply that feedback alignment, equilibrium propagation, and three-factor rules are equivalent mechanisms.",
        "transition": "That shared structure gives the standard local factorization.",
    },
    {
        "time": "0:50",
        "title": "Local learning preserves eligibility and approximates the returned signal",
        "purpose": "Define local learning and pre-empt the misconception that a shared scalar collapses neurons.",
        "say": (
            "For a point neuron, the exact weight gradient factorizes into a local eligibility "
            "and a neuron-specific loss signal. Eligibility uses quantities available at the "
            "connection: presynaptic activity and the postsynaptic derivative. A local rule "
            "can preserve this exact factor while replacing delta u by whatever learning "
            "signal is available. At one extreme every neuron receives the same scalar; at "
            "the other, feedback preserves one coordinate per neuron. A shared scalar does "
            "not make all weights equal, because every weight still has its own eligibility. "
            "It limits the rank of task-dependent learning information, so diversity remains "
            "but is underused."
        ),
        "read": "Read the factorization from left to right: local eligibility e i times exact neuron signal delta u; the available signal replaces only the second factor.",
        "question": "Would one shared error make the hidden layer behave like one neuron?",
        "answer": "No. Different inputs, activations, initial weights, and eligibilities keep units distinct. The limitation is feedback bandwidth: only one task-dependent direction is broadcast per example.",
        "guardrail": "Call this a bandwidth bottleneck, not neuronal collapse.",
        "transition": "A point neuron offers one destination; a dendritic neuron offers many internal destinations.",
    },
    {
        "time": "0:55",
        "title": "Dendrites turn one neuronal signal into four testable routing questions",
        "purpose": "Define the paper's organizing variables before any dendritic result is shown.",
        "say": (
            "We now replace one point unit by a neuron with N internal compartments. The "
            "available compartment field is A u times c u. The matrix A u specifies where "
            "each of K feedback channels is delivered; c u contains their signed values on "
            "the current example. This separates four questions that are often conflated. "
            "Coordinate asks whether feedback identifies the correct neuron. Ownership asks "
            "whether that coordinate reaches the correct arbor. Address asks which subtree "
            "within the arbor receives it. Gain asks how strongly the signal propagates. "
            "Finally, alignment asks whether these available routes span the credit pattern "
            "the task actually demands."
        ),
        "read": "A u has one row per compartment and one column per available channel. Multiplying by c u produces one available value at every compartment.",
        "question": "Is A u the anatomical adjacency matrix?",
        "answer": "No. It is a delivery or address matrix. Anatomy can generate its columns, but matched non-anatomical and deranged matrices let us separate route capacity from a specific topology.",
        "guardrail": "Grouped-point controls can emulate the same routing matrix; an address need not uniquely require dendritic material.",
        "transition": "To derive how route gain works, we need the conductance model.",
    },
    {
        "time": "1:05",
        "title": "Conductance makes dendritic voltage a normalized quotient",
        "purpose": "Define excitatory, inhibitory, additive, and shunting terms before deriving gradients.",
        "say": (
            "Each compartment receives nonnegative excitatory and inhibitory conductances. "
            "They are not distinguished by positive versus negative weights; they are "
            "distinguished by their reversal potentials. At steady state, voltage is the "
            "total reversal-weighted current divided by total conductance. Excitation and "
            "inhibition therefore affect the numerator, while both also increase the "
            "denominator and reduce local input resistance. Our additive control changes "
            "current without changing total conductance. A shunt can carry almost no net "
            "current when voltage is near the inhibitory reversal potential and still change "
            "the denominator. That denominator effect is what can regulate sensitivity and "
            "the gain of a credit route."
        ),
        "read": "G E and G I are sums of active synaptic conductances. R total is the reciprocal of leak, synaptic, and effective child conductance.",
        "question": "How can inhibition matter when its instantaneous current is near zero?",
        "answer": "Because conductance changes input resistance. Near the inhibitory reversal potential the numerator contribution can vanish while the denominator still increases.",
        "guardrail": "The additive comparison is a current-based control, not a claim that all point-neuron inhibition is additive.",
        "transition": "Differentiating this steady state reveals what a synapse can compute locally and what must be transported.",
    },
    {
        "time": "1:10",
        "title": "The exact dendritic gradient is local eligibility × transported compartment error",
        "purpose": "Present the central gradient factorization and distinguish directed-tree transport from the general adjoint.",
        "say": (
            "For conductance g i on compartment n, the exact gradient again has two factors. "
            "The local eligibility contains presynaptic activity, local input resistance, and "
            "the driving force between the synaptic reversal potential and local voltage. "
            "The second factor is delta V n u: how the loss changes if voltage at that "
            "compartment is perturbed. In the directed-tree approximation, this compartment "
            "error equals the somatic error times a path gain, alpha tilde n, which is a "
            "product of local derivatives, resistances, and coupling conductances along the "
            "unique path to the soma. For reciprocal cable models, the same error field is "
            "obtained by solving the steady-state adjoint."
        ),
        "read": "Read the first equation as local dendritic eligibility times transported compartment error. Alpha tilde n is dimensionless route gain along the directed path.",
        "question": "Does the path-product formula apply to a fully reciprocal dendrite?",
        "answer": "Not literally. It is exact for the directed-tree model. The more general result is the adjoint linear solve; the eligibility-times-compartment-error factorization remains valid.",
        "guardrail": "Do not present one directed path as the general Green's function of a reciprocal cable.",
        "transition": "We can now abstract any restricted feedback pathway as an operator on this exact field.",
    },
    {
        "time": "0:55",
        "title": "A feedback pathway selects, assigns, and scales stochastic credit",
        "purpose": "Introduce the common mathematical object used to compare feedback schemes.",
        "say": (
            "Let mu plus xi denote a stochastic exact gradient: mu is its mean task-aligned "
            "component and xi is zero-mean sampling noise. A feedback architecture applies "
            "an operator M before the update. Identity M gives unrestricted backpropagation. "
            "A scalar broadcast, one coordinate per neuron, subtree routes, deranged routes, "
            "and an exact compartment field are different choices of M. This abstraction "
            "separates the forward neuron from the information geometry of its learning "
            "signal. It also lets us ask one quantitative question across experiments: how "
            "much useful signal does a restricted route retain, and how much update energy "
            "and noise does it admit?"
        ),
        "read": "M acts on the stochastic gradient field. It can restrict span, reassign coordinates, or rescale them.",
        "question": "Is M learned by the model?",
        "answer": "Not necessarily. The theory describes any fixed or state-dependent delivery operator. In our controlled comparisons, M is prescribed so its effects can be isolated.",
        "guardrail": "This is a local linearization of delivered credit, not a claim that every biological pathway is globally linear.",
        "transition": "The signal-noise utility follows from the smoothness bound for one update.",
    },
    {
        "time": "1:05",
        "title": "Operator utility balances retained signal, update cost, and noise",
        "purpose": "Make the phase theory the predictive centerpiece and state its scope honestly.",
        "say": (
            "The numerator measures squared alignment between the true mean gradient and the "
            "routed update. The denominator penalizes both the size of the retained signal "
            "and the routed noise, scaled by curvature. So a restricted operator can help "
            "when it preserves aligned signal while rejecting noisy or irrelevant "
            "dimensions. We evaluated this quantity before training across 540 conditions. "
            "Its rank correlation is 0.937 with observed norm-matched one-step progress and "
            "0.916 with final accuracy. This is the paper's strongest quantitative result: "
            "one operator theory predicts major wins, nulls, and crossovers. It is not a "
            "guarantee for every small difference accumulated over a nonlinear trajectory."
        ),
        "read": "Read U of M as aligned signal squared divided by curvature times routed signal-plus-noise energy.",
        "question": "Does a high U guarantee higher final test accuracy?",
        "answer": "No. It is exact for an isotropic quadratic and otherwise a one-step smoothness guarantee. The strong final-accuracy association is empirical validation, not a theorem about all training trajectories.",
        "guardrail": "Lead with prediction of large contrasts and phase boundaries; keep trajectory-scale exceptions explicit.",
        "transition": "The first boundary appears on ordinary image tasks.",
    },
    {
        "time": "1:00",
        "title": "Neuron-specific feedback dominates standard image tasks",
        "purpose": "Establish that neuron identity, rather than exact dendritic address, is the main bottleneck on standard benchmarks.",
        "say": (
            "These ladders progressively increase feedback resolution. Moving from one strict "
            "layer-wide scalar to one coordinate per neuron improves accuracy by about eight "
            "to sixteen percentage points across MNIST and flattened CIFAR-10. Moving from "
            "one neuron-specific coordinate to the exact compartment field changes accuracy "
            "by less than one point. On CIFAR-10, the exact field is within the predefined "
            "one-point equivalence margin of backpropagation. The important conclusion is a "
            "boundary, not a universal dendritic benefit: on standard image tasks, preserving "
            "which neuron owns a learning signal is much more important than resolving the "
            "exact path inside that neuron."
        ),
        "read": "Compare adjacent rungs, not absolute axes across datasets. Green denotes shunting on MNIST; blue denotes the additive tree.",
        "question": "Why does the scalar network still learn well?",
        "answer": "Every weight retains a distinct eligibility and the forward network retains many hidden units. Scalar feedback restricts task-dependent credit but does not erase representational diversity.",
        "guardrail": "Do not claim exact dendritic routing is generally needed from these benchmarks; they show the opposite.",
        "transition": "To make a branch address necessary, the task must demand incompatible updates inside one neuron.",
    },
    {
        "time": "1:00",
        "title": "Credit conflict creates a demand for branch-specific signals",
        "purpose": "Define the controlled branch-conflict task before showing its outcome.",
        "say": (
            "Here every branch is active and therefore has nonzero eligibility, but a context "
            "selects which branch determines the label. The conflict parameter chi controls "
            "the distractors. At chi equals zero, selected and nonselected streams are "
            "class-compatible, so one shared learning signal can work. At chi equals one, "
            "the nonselected branches carry the opposite class, so simultaneously active "
            "branches require different or opposing updates. The branch-specific rule gates "
            "credit by context; the shared rule broadcasts the same value across branches. "
            "This is a deliberately constructed existence test of when within-neuron address "
            "resolution is required."
        ),
        "read": "B is the number of branches, c t selects the relevant branch, and d b is the mean update direction assigned to branch b.",
        "question": "Was this task designed to favor branch-specific routing?",
        "answer": "Yes, deliberately. It is a mechanism-matched necessity test: we vary the precise feature, simultaneous within-neuron credit conflict, that the theory says should create path demand.",
        "guardrail": "Call it a controlled existence proof, not a naturalistic benchmark.",
        "transition": "The theory predicts the exact conflict value at which shared credit changes sign.",
    },
    {
        "time": "0:55",
        "title": "Shared credit fails at the predicted conflict boundary",
        "purpose": "Show that the analytic crossover predicts trained failure and that route assignment is causal.",
        "say": (
            "The shared update has an analytic eigenvalue B minus two chi times B minus one. "
            "It crosses zero at chi c equals B over two times B minus one. The trained shared "
            "rule collapses near this predicted boundary for two, four, and eight branches. "
            "At full conflict, branch-specific routing gains thirty-five to fifty-eight "
            "percentage points. A cyclic derangement keeps the same rank and sparsity but "
            "delivers each channel to the wrong branch, and it fails. Analytic "
            "backpropagation, the correct routed field, and a gated-point implementation "
            "coincide because they are equivalent calculations of the same routing matrix."
        ),
        "read": "Lambda shared is the useful shared-mode coefficient. Its sign change defines chi c; the plotted collapse points track that boundary.",
        "question": "Why does the critical conflict depend on B?",
        "answer": "One selected stream contributes useful signal while B minus one distractors contribute conflicting signal. Increasing B changes that balance and shifts the zero crossing.",
        "guardrail": "The result establishes the need for an address, not a unique material need for a dendritic tree.",
        "transition": "A second task asks whether a small hierarchy of addresses can be efficient.",
    },
    {
        "time": "0:50",
        "title": "Nested tasks ask whether a few subtree addresses are efficient",
        "purpose": "Define the hierarchy task and make clear that K is feedback bandwidth.",
        "say": (
            "The next task has eight input streams organized by a nested context hierarchy. "
            "We expose only K independent feedback channels inside each neuron: one shared "
            "channel, two coarse subtrees, four intermediate subtrees, or eight leaf-specific "
            "channels. A K therefore has rank K. We compare true ancestry routes with "
            "derangements and matched non-anatomical low-rank bases while holding forward "
            "resources, rank, sparsity, and parameter count fixed. K is feedback bandwidth, "
            "not physical dendritic depth. The hypothesis is that a tree becomes an efficient "
            "basis only when the task's credit covariance has a compatible nested structure."
        ),
        "read": "The four route diagrams increase within-neuron bandwidth K from one to full rank eight.",
        "question": "Is increasing K the same as adding dendritic depth?",
        "answer": "No. K counts independent returned coordinates. Physical depth changes the forward serial computation and is a separate experiment kept in backup.",
        "guardrail": "Do not let the audience interpret K as number of dendritic levels.",
        "transition": "The result separates address ownership from the smaller contribution of this particular topology.",
    },
    {
        "time": "0:55",
        "title": "Subtree addresses help—but fine topology adds only a narrow gain",
        "purpose": "Separate the large value of correct address assignment from the small topology-specific effect.",
        "say": (
            "At intermediate bandwidth K equals four, assigning the channels to the correct "
            "subtrees beats a cyclic derangement by sixty-one percentage points. That is the "
            "large address-assignment effect. But against the strongest matched "
            "non-anatomical low-rank basis, the ancestry advantage is only 1.27 points. "
            "Ancestry loses at K equals one and two, wins at K equals four, and ties at full "
            "rank. Rewiring removes the intermediate gain. Dendritic, grouped-point, and "
            "gated-point implementations coincide when they receive the same routed field. "
            "So the robust conclusion is that correct addresses matter; the extra value of "
            "nested anatomy is narrow and task-matched."
        ),
        "read": "Compare correct ancestry first with derangement, then with the best rank- and sparsity-matched alternative. Those contrasts answer different questions.",
        "question": "Does the sixty-one-point effect prove that dendritic topology is superior?",
        "answer": "No. It proves that correct channel-to-subtree assignment matters. The topology-specific comparison is the much smaller 1.27-point advantage over the best matched alternative at K equals four.",
        "guardrail": "Always pair the large assignment effect with the modest topology-specific effect.",
        "transition": "If biological trees are candidate route dictionaries, how much field can real arbors represent per connection?",
    },
    {
        "time": "0:55",
        "title": "Real arbors provide sparse candidate routes, mostly through coarse geometry",
        "purpose": "Present anatomy as route capacity, not evidence of biological use.",
        "say": (
            "We map excitatory and inhibitory contacts onto reconstructed MICrONS arbors and "
            "use branch points to define nested route supports. Capture is the fraction of a "
            "target field's energy that lies in the route span. These sparse dictionaries "
            "retain about eighty-five percent of dense capture using about seven percent of "
            "the dense route-matrix connections. That is a 14.2-fold dense-normalized ratio, "
            "but the anatomy-specific advantage over a density-matched shuffled dictionary "
            "is closer to 2.7-fold. The ordering replicates in held-out cells and a small "
            "second-mouse cohort. Most of the capacity is explained by coarse branch geometry."
        ),
        "read": "P A projects a field q into the span of the route matrix A. Capture ranges from zero to one.",
        "question": "Does high capture show that the animal uses these routes for learning?",
        "answer": "No. It shows representational capacity of modeled route dictionaries. It does not measure endogenous task gradients or demonstrate plasticity through those routes.",
        "guardrail": "Pair 14.2-fold with the density-matched approximately 2.7-fold control and call the second-mouse result descriptive.",
        "transition": "Anatomy supplies addresses; conductance can in principle regulate their gain.",
    },
    {
        "time": "1:00",
        "title": "Focal shunting changes descendant credit only in permissive regimes",
        "purpose": "Explain the mechanistic contrast and foreground the standard-calibration null.",
        "say": (
            "We compare a focal inhibitory conductance with an additive current chosen to "
            "match the baseline first-order focal current, then restore somatic voltage. The "
            "local dendritic voltage is intentionally not matched, because the question is "
            "whether conductance changes sensitivity beyond current injection. In the "
            "directed tree, the log gain of a descendant route changes in proportion to minus "
            "the local input resistance when the shunt lies on that route. At the standard "
            "passive calibration the localization contrast is essentially zero. In "
            "high-conductance, electrotonically permissive regimes, descendant-localized "
            "changes emerge and survive active-channel extensions. Shunting is therefore a "
            "conditional gain mechanism, not a generic learning improvement."
        ),
        "read": "The indicator is one only for descendants of compartment k, and the effect scales with R total k.",
        "question": "Is the main shunting result positive or null?",
        "answer": "At the standard passive calibration it is null. The positive localization appears in a defined permissive high-conductance regime, which is why the claim is conditional.",
        "guardrail": "Do not imply that shunting improves training generically or that local voltage was matched.",
        "transition": "Capacity and conditional gain still do not show that measured cortical function uses these routes.",
    },
    {
        "time": "0:50",
        "title": "Measured visual responses show no morphology-specific alignment",
        "purpose": "State the biological null as a headline result and define its inferential scope.",
        "say": (
            "For seven reconstructed target cells from one MICrONS mouse, we map measured "
            "presynaptic visual responses through the conductance model and test held-out "
            "postsynaptic-response prediction. Nested subtrees do not outperform random or "
            "site-shuffled route dictionaries for held-out learning, field capture, or "
            "within-arbor structure-function similarity. The effects are centered near zero. "
            "This is an important boundary: measured anatomy offers possible addresses, but "
            "these visual-response data do not reveal preferential alignment between those "
            "addresses and the functional relation being modeled. The cohort is small and "
            "does not measure plasticity, so the result is a scoped null rather than a general "
            "rejection of dendritic learning."
        ),
        "read": "The pipeline moves from measured presynaptic responses to mapped conductances and then held-out postsynaptic prediction; the contrast plot compares topology with matched controls.",
        "question": "Does this rule out morphology-specific credit assignment in cortex?",
        "answer": "No. It rules out a detectable advantage in this seven-cell, one-mouse visual-response analysis. Other tasks, states, cell types, or direct plasticity measurements could differ.",
        "guardrail": "Say seven cells and one mouse aloud; never generalize the null beyond this cohort and assay.",
        "transition": "We therefore ask the narrower causal question: would the same routes help if the task field were aligned to them?",
    },
    {
        "time": "0:50",
        "title": "The same anatomical routes capture task credit aligned to their span",
        "purpose": "Show controlled sufficiency of alignment without implying endogenous biological use.",
        "say": (
            "We construct a fixed-energy field phi of a by rotating between a unit vector "
            "inside the anatomical route span and an orthogonal unit vector. Anatomy, field "
            "energy, curvature, and route count remain fixed; only alignment a changes. By "
            "construction, anatomical capture increases linearly with a, and the true "
            "subtree routes outperform matched controls as the field enters their span. This "
            "rescue shows that the anatomical dictionary is sufficient to represent a task "
            "field when the geometry is matched. It does not show that alignment was learned, "
            "that these are endogenous cortical error signals, or that the manipulation "
            "improves a trained network."
        ),
        "read": "u parallel lies in the column space of A and u perpendicular is orthogonal to it. The square roots keep total field energy fixed.",
        "question": "Is this a trained-learning experiment?",
        "answer": "No. It is a controlled representational analysis. Its purpose is to isolate alignment as the missing variable after the measured-response null.",
        "guardrail": "Use the phrase conditional representational sufficiency.",
        "transition": "The standard tasks, controlled positives, and biological nulls now occupy one common phase plane.",
    },
    {
        "time": "0:45",
        "title": "One alignment–bandwidth plane organizes the wins and nulls",
        "purpose": "Give the audience one unifying map rather than a list of disconnected results.",
        "say": (
            "The horizontal axis is task-route alignment: how much demanded credit lies in "
            "the available route span. The vertical axis is feedback bandwidth relative to "
            "the effective dimensionality of the task-credit field. Low bandwidth creates a "
            "coordinate or address bottleneck. At aligned intermediate bandwidth, restricted "
            "routes can preserve useful signal while rejecting irrelevant dimensions. At "
            "full rank, route capacity saturates; at weak alignment, extra structure cannot "
            "help. Each experiment estimates its coordinates separately. The background "
            "regimes and the bandwidth-equals-effective-rank boundary are theoretical, not "
            "fitted to the outcome points."
        ),
        "read": "Move right for stronger task-route alignment and up for more independent feedback coordinates relative to effective credit rank.",
        "question": "Was the phase boundary fitted to these experiments?",
        "answer": "No. The regime tint and K over effective-rank equals one line come from the operator theory. Experimental coordinates are estimated independently within each assay.",
        "guardrail": "Do not treat approximate point placement as a universal calibrated phase diagram.",
        "transition": "This leaves four conclusions, ordered by what the evidence establishes most strongly.",
    },
    {
        "time": "0:45",
        "title": "Coordinate → address → gain, all conditional on task–route alignment",
        "purpose": "End with four defensible statements and one memorable novelty claim.",
        "say": (
            "The first requirement is neuron-specific coordinate: it closes most of the "
            "scalar-to-exact gap on standard tasks. Within-neuron addresses become necessary "
            "when simultaneously active branches require conflicting updates. Nested "
            "topology adds a modest advantage only at matched intermediate bandwidth, while "
            "real arbors provide sparse candidate routes. Conductance can regulate route gain "
            "in permissive states, but measured endogenous use remains unestablished. The "
            "central contribution is therefore not that dendrites always improve learning. "
            "It is a predictive boundary map of when restricted dendritic routes should help, "
            "when they should not, and why."
        ),
        "read": "The final flow is coordinate, address, and gain, gated jointly by alignment and bandwidth.",
        "question": "What is the one-sentence novelty?",
        "answer": "A stochastic credit-operator theory quantitatively predicts when restricted dendritic routes preserve useful learning signal, and controlled experiments map both its positive regimes and its null boundaries.",
        "guardrail": "Finish on the conditional theory; do not inflate anatomy or shunting into universal benefits.",
        "transition": "Thank the audience and invite questions.",
    },
]


STYLE = """
@page { size: 11in 8.5in; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; color: #12233F; font-family: Arial, sans-serif; }
.page {
  width: 11in; height: 8.5in; page-break-after: always; overflow: hidden;
  padding: .34in .38in .30in; background:
    radial-gradient(circle at 100% 0%, rgba(22,143,131,.07), transparent 28%),
    linear-gradient(180deg,#FBFBF8,#F5F7F4);
}
.head { display: grid; grid-template-columns: .65in 1fr 1.05in; gap: .1in;
  align-items: end; padding-bottom: .11in; border-bottom: 2px solid #CADDD9; }
.num { color: #168F83; font: 700 25px Georgia, serif; }
h1 { margin: 0; font: 700 23px/1.07 Georgia, serif; letter-spacing: -.2px; }
.time { color: #168F83; font-weight: 800; font-size: 15px; text-align: right; }
.body { display: grid; grid-template-columns: 31.5% 68.5%; gap: .24in; margin-top: .18in; }
.left, .right { min-width: 0; }
.thumb { width: 100%; border: 1px solid #CBD6DD; border-radius: 8px; box-shadow: 0 5px 16px rgba(18,35,63,.10); }
.block { margin-top: .12in; padding: .10in .12in; border-radius: 8px; background: #FFFFFF; border-left: 4px solid #168F83; }
.block.purple { border-left-color: #7654B5; }
.block.orange { border-left-color: #E48743; }
.block.gray { border-left-color: #8B98A7; }
.label { color: #168F83; font-size: 9px; font-weight: 800; letter-spacing: .8px; text-transform: uppercase; margin-bottom: 3px; }
.purple .label { color: #7654B5; } .orange .label { color: #B9692E; } .gray .label { color: #637181; }
p { margin: 0; font-size: 10.5px; line-height: 1.34; }
.script { padding: .15in .18in; border: 1.5px solid #C9E3DE; border-radius: 10px; background: #EFF8F6; }
.script p { font-size: 12.2px; line-height: 1.38; }
.qa { display: grid; grid-template-columns: 43% 57%; gap: .12in; margin-top: .13in; }
.qa .block { margin-top: 0; }
.transition { margin-top: .13in; padding: .10in .14in; border-radius: 8px; background: #12233F; color: white; }
.transition .label { color: #A8DCD4; }
.footer { position: absolute; left: .38in; right: .38in; bottom: .12in; color: #73808E; font-size: 8px; text-align: right; }
"""


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def build_markdown() -> None:
    lines = [
        "# When can dendritic structure help local credit assignment?",
        "",
        "## 20-minute workshop speaker notes",
        "",
        "Scripted time: 17:55. The remaining 2:05 is reserved for pauses, emphasis, and slide transitions.",
        "",
    ]
    for index, note in enumerate(NOTES, start=1):
        lines.extend([
            f"## {index}. {note['title']} [{note['time']}]",
            "",
            f"**Purpose.** {note['purpose']}",
            "",
            f"**Say.** {note['say']}",
            "",
            f"**How to read the slide.** {note['read']}",
            "",
            f"**Likely question.** {note['question']}",
            "",
            f"**Answer.** {note['answer']}",
            "",
            f"**Claim guardrail.** {note['guardrail']}",
            "",
            f"**Transition.** {note['transition']}",
            "",
        ])
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")


def build_html() -> None:
    pages = []
    total = len(NOTES)
    for index, note in enumerate(NOTES, start=1):
        thumbnail = PNG_DIR / f"slide_{index:02d}.png"
        if not thumbnail.exists():
            raise FileNotFoundError(thumbnail)
        pages.append(f"""
<section class="page">
  <header class="head">
    <div class="num">{index:02d}</div>
    <h1>{esc(note["title"])}</h1>
    <div class="time">{esc(note["time"])} · scripted</div>
  </header>
  <div class="body">
    <aside class="left">
      <img class="thumb" src="{thumbnail.as_uri()}" alt="Slide {index} thumbnail">
      <div class="block"><div class="label">Purpose</div><p>{esc(note["purpose"])}</p></div>
      <div class="block purple"><div class="label">How to read it</div><p>{esc(note["read"])}</p></div>
      <div class="block orange"><div class="label">Claim guardrail</div><p>{esc(note["guardrail"])}</p></div>
    </aside>
    <main class="right">
      <div class="script"><div class="label">Say this</div><p>{esc(note["say"])}</p></div>
      <div class="qa">
        <div class="block purple"><div class="label">Likely question</div><p>{esc(note["question"])}</p></div>
        <div class="block"><div class="label">Answer</div><p>{esc(note["answer"])}</p></div>
      </div>
      <div class="transition"><div class="label">Transition</div><p>{esc(note["transition"])}</p></div>
    </main>
  </div>
  <div class="footer">Kempner Learning Dynamics Workshop · slide {index} of {total}</div>
</section>""")
    document = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<style>{STYLE}</style></head><body>{''.join(pages)}</body></html>"
    )
    HTML_OUT.write_text(document, encoding="utf-8")


def build_pdf() -> None:
    chromium = shutil.which("chromium-browser") or shutil.which("chromium")
    if chromium is None:
        raise SystemExit("Chromium is required to render the speaker guide")
    command = [
        chromium,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--allow-file-access-from-files",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={PDF_OUT}",
        HTML_OUT.as_uri(),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise SystemExit(result.stderr or result.stdout)


def validate() -> None:
    if len(NOTES) != 20:
        raise SystemExit(f"Expected 20 notes, found {len(NOTES)}")
    with fitz.open(PDF_OUT) as document:
        if document.page_count != len(NOTES):
            raise SystemExit(f"Speaker guide has {document.page_count} pages")
        for index, page in enumerate(document, start=1):
            if page.rect.width <= page.rect.height:
                raise SystemExit(f"Guide page {index} is not landscape")
            if len(page.get_text().strip()) < 500:
                raise SystemExit(f"Guide page {index} has unexpectedly little text")
    print(f"Validated {len(NOTES)} speaker-guide pages; scripted time 17:55")


def main() -> None:
    build_markdown()
    build_html()
    build_pdf()
    validate()


if __name__ == "__main__":
    main()
