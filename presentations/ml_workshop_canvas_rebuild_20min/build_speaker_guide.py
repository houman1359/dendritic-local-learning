#!/usr/bin/env python3
"""Build the narrated guide for the 25-slide, 20-minute workshop deck."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
LEGACY_BUILDER = HERE.parent / "ml_workshop_15_20min" / "build_speaker_guide.py"


def note(time: str, title: str, purpose: str, say: str, read: str,
         question: str, answer: str, guardrail: str, transition: str) -> dict[str, str]:
    return {
        "time": time,
        "title": title,
        "purpose": purpose,
        "say": say,
        "read": read,
        "question": question,
        "answer": answer,
        "guardrail": guardrail,
        "transition": transition,
    }


NOTES = [
    note(
        "0:15",
        "When dendritic structure helps local credit assignment",
        "State the conditional scientific question and the answer the talk will defend.",
        "A network-level error must ultimately change internal weights. Backpropagation gives the exact computational reference, but a biological circuit must learn with signals available to neurons and their dendrites. I will ask when dendritic structure gives those local signals a useful coordinate system. The answer is conditional: dendrites help when the task-relevant credit field aligns with the available routes, the returned bandwidth is appropriate, and transport is reliable.",
        "Read the graphic from the network loss to a neuron, then along its arbor to a route-aligned credit field. The subspace diagram is the destination of the story, not a decorative metaphor.",
        "Are you claiming that dendrites implement backpropagation?",
        "No. Backpropagation defines the exact information that learning would require. The work studies what is retained or lost when biological feedback is spatially restricted.",
        "Use conditional language throughout; do not promise a general dendritic advantage.",
        "I will organize the evidence around four separable questions.",
    ),
    note(
        "0:30",
        "Four separable questions organize the evidence",
        "Give the audience a map before introducing models or experiments.",
        "The talk separates four issues that are often conflated. Neuron specificity asks which neurons can receive distinct learning signals. Dendritic location asks where inside one neuron a signal should act. Conductance asks how strongly it is transmitted. Task match asks whether the task actually requires the routes the neuron supplies. We move from an exact derivation, to a predictive operator theory, to controlled tasks, reconstructed anatomy, and finally measured functional data.",
        "Scan the four columns from coordinate to alignment, then follow the lower evidence path. Each later experiment answers one stated question rather than adding an unrelated benchmark.",
        "Are coordinate, address, gain, and alignment four sequential biological stages?",
        "No. They are analytically separable properties of a learning signal and its delivery. A biological mechanism may couple them.",
        "Keep route availability, mechanistic sufficiency, and endogenous biological use distinct.",
        "First we need to define credit assignment at the level familiar to an ML audience.",
    ),
    note(
        "0:35",
        "A network objective does not identify its local causes",
        "Define structural credit assignment in ordinary neural-network language.",
        "Training minimizes a network objective by changing many internal weights. The global loss tells us whether the output was wrong, but not which hidden weight caused that error, in which direction it should move, or by how much. Credit assignment is the calculation of those parameter-specific derivatives. In a biological network the same question can be asked at several nested scales: which neuron, which branch within that neuron, and which weight or synapse on that branch.",
        "Follow the single loss backward toward multiple highlighted weights. The equation says that each weight has its own derivative even though all weights contribute to one objective.",
        "Is this temporal credit assignment?",
        "No. This talk focuses on structural credit across neurons and dendritic locations. Temporal eligibility traces could be combined with the same framework.",
        "Begin with network weights and objectives; introduce synaptic implementation only after the computational problem is clear.",
        "Backpropagation supplies the exact computational solution and therefore our reference signal.",
    ),
    note(
        "0:35",
        "Backpropagation provides the exact computational reference",
        "Establish the eligibility-times-error factorization that is retained throughout the talk.",
        "For a hidden unit, the exact gradient of weight w_ij factorizes into the presynaptic activity x_j and the hidden-node sensitivity delta_i. Backpropagation computes delta_i by applying the chain rule through downstream weights and nonlinearities. The x_j term is locally available eligibility; delta_i carries the task-dependent downstream consequence. This exact signal is an information ceiling, not a claim about a biological reverse pass.",
        "Blue arrows show the forward computation and purple arrows the reverse derivative. Read the equation from left to right as weight gradient equals local eligibility times transported error.",
        "Why use backpropagation in a paper about biological learning?",
        "Because it defines the correct descent direction against which the information content of restricted local signals can be evaluated.",
        "Do not say that backpropagation sends a separate message to every weight; parameter specificity also comes from local eligibility.",
        "Local-learning theories preserve this factorization but change the source and spatial resolution of the returned signal.",
    ),
    note(
        "0:40",
        "From BP to local to dendritic-local learning",
        "Make the requested conceptual map explicit and place the work relative to prior approaches.",
        "The left column is exact backpropagation: eligibility is multiplied by the exact node error. A point-neuron local rule keeps the same structure but replaces that error by an approximate signal available at the neuron. A dendritic-local rule further asks which compartment receives which component of the signal. Prior work differs in how the approximate signal is generated—three-factor modulation, feedback alignment, equilibrium or predictive dynamics, and dendritic error segregation. Our contribution is about the coordinate and destination of that signal once it is available.",
        "Compare the three equations horizontally. The local factor remains on every column; the returned factor becomes increasingly restricted from network node to neuron to dendritic location.",
        "Is the dendritic rule a new alternative to every existing local-learning algorithm?",
        "No. It is a delivery and representation framework that can in principle sit on top of different mechanisms that generate an approximate learning signal.",
        "Do not present the cited approaches as equivalent or as one strict historical sequence.",
        "The first restriction is more basic than dendritic location: can feedback identify the correct neuron?",
    ),
    note(
        "0:45",
        "Neuron identity is the first feedback bottleneck",
        "Explain why scalar feedback underuses a layer without collapsing all neurons into one.",
        "A layer-wide scalar multiplies every neuron’s own eligibility. The neurons do not collapse: their inputs, voltages, nonlinear derivatives, and initial weights remain different. What is compressed is the task-dependent part of the update, which has only one shared coordinate. Giving one signed value per neuron restores neuronal identity. Only after that bottleneck is closed does it make sense to ask whether different locations within one arbor need different values.",
        "Move from the scalar broadcast on the left, to neuron-specific coordinates in the center, to opposing within-neuron branch signals on the right.",
        "Would a shared scalar make every neuron learn the same weights?",
        "No. Each update is the shared scalar times a neuron- and synapse-specific eligibility. The condition limits feedback bandwidth rather than erasing forward-state diversity.",
        "The scalar-versus-neuronal bandwidth issue is established prior work; the novel question begins with ownership and within-arbor address.",
        "Dendrites provide candidate spatial resources for resolving that next level.",
    ),
    note(
        "0:35",
        "A dendritic tree adds local state, subtree address, and route gain",
        "Introduce what a dendritic model adds beyond a point neuron before claiming any benefit.",
        "A dendritic tree introduces a field of compartment voltages and conductances, so local eligibility can differ across locations. Its branching structure supplies nested subtree supports that could serve as addresses. Conductance and electrotonic attenuation can regulate how strongly a signal propagates along those routes. These are candidate resources: the remainder of the talk asks whether they are needed, whether they are sufficient, and whether measured biological signals use them.",
        "Read the central arbor together with the four direct labels: local state, subtree address, route gain, and task alignment. The last item is a requirement imposed by the task, not a structural property of the tree.",
        "Does the existence of branches imply better local learning?",
        "No. Structure creates possible coordinates and operations. A benefit requires the task-credit field to match them.",
        "Treat these as available resources, not evidence of endogenous learning use.",
        "To derive a local rule, we first need a conductance model that distinguishes excitation from inhibition.",
    ),
    note(
        "0:50",
        "Excitation and inhibition differ through reversal potential, while both load the membrane",
        "Define the steady-state dendritic neuron and make shunting explicit.",
        "Excitatory and inhibitory synapses are represented by nonnegative conductances. Their currents differ through the driving force E_X minus V_n: excitation pulls voltage toward E_E and inhibition toward E_I. Both conductances also enter the total conductance in the denominator, reduce input resistance, and thereby change the influence of every other current. That divisive loading is the shunting component. A matched negative additive current can reproduce a baseline voltage change but not this change in the conductance operator.",
        "Use the voltage axis to distinguish E_E, V_n, and E_I. In the steady-state fraction, point to the reversal-potential terms in the numerator and the shared conductance loading in the denominator.",
        "Is inhibition simply a negative weight in these equations?",
        "No. Sign is set by the state-dependent driving force, and an inhibitory conductance also increases total conductance. That denominator effect is absent from an additive-current control.",
        "The reversal potentials are model conventions for the rate regime, not fitted intracellular millivolt values.",
        "Differentiating this steady state gives the exact local eligibility for both excitatory and inhibitory conductances.",
    ),
    note(
        "0:45",
        "The exact conductance gradient separates local eligibility from compartment error",
        "Derive the central dendritic-gradient factorization and define the excitatory and inhibitory cases.",
        "For a conductance g_i^X on compartment n, the exact gradient is presynaptic activity times total input resistance times the local driving force, multiplied by the loss derivative with respect to that compartment voltage. The first three factors are local eligibility. Excitation and inhibition differ through E_X minus V_n. The unresolved factor is delta_n^V: how changing voltage at this particular compartment would change the network loss.",
        "Read the colored brace as local eligibility and the remaining factor as transported compartment error. Compare E_E minus V_n with E_I minus V_n to see where excitatory and inhibitory signs enter.",
        "Does this factorization make the exact gradient biologically local?",
        "Not by itself. Eligibility is local, but the compartment error contains downstream task information and still has to be transported or approximated.",
        "Never call eligibility alone the gradient; the task-dependent compartment factor is essential.",
        "We next derive how that compartment error is transported through a tree.",
    ),
    note(
        "0:40",
        "Tree transport is a special case of the general adjoint",
        "Distinguish the intuitive directed path product from exact transport in reciprocal or active arbors.",
        "In a feedforward directed tree, a compartment’s error is the somatic error multiplied by local transfer derivatives along its unique ancestry path. That gives a useful path-gain intuition. A reciprocal conductance tree is not a one-way path, however. The general exact calculation solves the steady-state adjoint equation J_V transpose q equals the voltage derivative of the loss. The component q_n tells how the loss changes under a small current perturbation at compartment n.",
        "Read the left side as the directed special case and the right side as the general reciprocal calculation. Both produce a spatial field over compartments, but only the left admits a unique path product.",
        "Is exact path feedback valid for a biophysical reciprocal cable?",
        "Only as a directed-tree special case or an intuition. The reciprocal conductance model uses the adjoint field.",
        "Do not use single-path language when discussing the general reciprocal or active model.",
        "With the exact field defined, we can state what a restricted local feedback system is allowed to deliver.",
    ),
    note(
        "0:40",
        "Restricted local learning is a routed credit operator",
        "Define the address matrix, communicated coefficients, and feedback bandwidth without notation ambiguity.",
        "For neuron u, the approximate compartment field is A_u c_u. A_u has one row per nonsomatic compartment and one column per available route. Each column specifies where one channel acts; c_u contains the example-dependent signed values carried by those channels. K is therefore feedback bandwidth. K equals one for a shared field, intermediate K gives partial subtree addresses, and full rank can represent every compartment coordinate. A_u is an address matrix, not the dendritic adjacency matrix.",
        "Use the route-support drawings to connect columns of A_u to locations in the arbor. Then read the equation as route geometry multiplied by communicated values.",
        "Where do the coefficients c_u come from?",
        "The framework does not require one unique generator. They can be supplied by feedback alignment, recurrent inference, modulatory pathways, or another local-credit mechanism.",
        "Keep feedback content, neuron-to-arbor ownership, and within-arbor destination conceptually separate.",
        "The next question is when restricting this field can improve rather than merely approximate learning.",
    ),
    note(
        "1:05",
        "Credit-operator utility predicts a conditional phase",
        "Present the paper’s central theory and its quantitative validation.",
        "The controlled validation uses a fixed-forward 16-coordinate Haar task. H_c denotes task-credit hierarchy and D_r feedback-route resolution. Let the exact stochastic update have mean mu and noise xi with covariance Sigma. A credit operator M retains and reweights part of that field. Utility compares aligned retained signal with routed update energy and admitted variance. This predicts alignment dependence, an intermediate-bandwidth optimum, a signal-to-noise boundary, and reliability-matched gain. Across 540 controlled conditions, utility ranks one-step progress at rho 0.937 and final outcomes at rho 0.916.",
        "First parse the numerator as useful descent signal and the denominator as update energy plus admitted noise. Then use the plot to connect the analytic quantity to observed optimization.",
        "Does this bound predict every difference after nonlinear training?",
        "No. It is exact for the stated quadratic case and otherwise a smoothness-based one-step guarantee. It predicts large contrasts, boundaries, and within-family ordering rather than every trajectory-accrued difference.",
        "Present the theory as the advance and the experiments as tests of its predicted boundary.",
        "We now test progressively harder demands, beginning with ordinary image classification.",
    ),
    note(
        "0:30",
        "Standard benchmarks vary feedback resolution while holding the forward model fixed",
        "Define the benchmark ladder and its controls before presenting accuracy.",
        "MNIST, Fashion-MNIST, and flattened CIFAR-10 are passed through matched tree architectures. The forward model, parameter count, optimizer, and training protocol are held fixed while feedback changes from one strict layer-wide scalar, to one coordinate per neuron repeated through its arbor, to the exact compartment field. Shunting and raw-additive trees have the same topology; the additive control removes reversal-potential and conductance-denominator effects.",
        "Read each dataset row into the same architecture, then move down the three feedback conditions. The scientific variable is feedback resolution, not dataset aesthetics.",
        "Why use flattened CIFAR-10 instead of a convolutional model?",
        "It is a harder controlled nonconvolutional task for the same credit-assignment ladder, not an attempt to report state-of-the-art vision performance.",
        "Do not infer that dataset difficulty alone determines whether path resolution helps.",
        "The result separates the need for neuron identity from the need for exact within-arbor paths.",
    ),
    note(
        "0:50",
        "Neuron identity closes almost the entire standard-task gap",
        "State the principal benchmark result and the important path-variation null.",
        "On MNIST, moving from a strict scalar to neuron-specific feedback improves shunting trees by 11.2 percentage points and raw-additive trees by 7.8. Exact path feedback adds less than 0.2 points. On flattened CIFAR-10, the confirmatory means are 34.47 percent for strict scalar, 50.87 for neuron-specific, 50.01 for exact path, and 50.12 for matched backpropagation. Path-specific distal error is nevertheless present—about 53 percent under shunting versus 32 percent under raw addition—but these tasks do not reward resolving it.",
        "Read the main accuracy ladder first. Then use the path-variation panel to separate the existence of within-arbor gradient structure from its usefulness on these tasks.",
        "Does the exact-path null mean dendritic paths carry no distinct information?",
        "No. The gradients vary across paths, especially under shunting. The null says that ordinary classification performance is already saturated once the correct neuron receives the signal.",
        "Do not generalize this benchmark null to tasks that require opposing branch-specific updates.",
        "The next task deliberately creates that missing demand.",
    ),
    note(
        "0:45",
        "Branch conflict creates a controlled need for an internal address",
        "Define the context-gated conflict task and its analytic crossing before showing trained outcomes.",
        "Each trial activates B branches. A context variable marks the branch whose image view supports the target; nonselected branches carry opposite-class evidence with probability chi. A branch-specific signal can update the selected branch correctly. A neuron-shared signal averages incompatible branch demands. Its alignment eigenvalue is B minus two chi times B minus one, so it changes sign at chi_c equal to B over two times B minus one. We compare correct routing with a cyclic derangement that preserves route rank and sparsity but sends signals to the wrong branches.",
        "Follow the selected branch marked by the c badge, then compare branch-specific and shared delivery. The lower equation predicts exactly where the shared descent direction disappears.",
        "Is this intended as a naturalistic vision benchmark?",
        "No. It is a mechanism-matched existence test that isolates when an internal branch address is mathematically necessary.",
        "State that all B branches are active and that x_y and x_1−y are class-conditioned views, not target labels attached to branches.",
        "Training tests whether the analytic sign change predicts the empirical failure boundary.",
    ),
    note(
        "0:45",
        "Shared credit fails at the predicted conflict boundary",
        "Show that branch ownership, rather than route rank alone, determines success under conflict.",
        "As conflict increases, the shared rule crosses from useful to harmful near the predicted chi_c for B equal to 2, 4, and 8. At full conflict, correct branch routing gains roughly 35 to 58 percentage points. The branch-by-conflict interaction is positive in all 20 seeds at every B. Cyclic derangement fails despite preserving rank and sparsity. The exact-route curves coincide because they are algebraically equivalent calculations of the same routing matrix, not independent algorithms.",
        "Read the accuracy curves from low to high conflict and compare the vertical crossings with the analytic prediction. Then inspect the deranged control to isolate destination from bandwidth.",
        "Does this prove that biological dendrites are uniquely required?",
        "No. It proves that branch-specific information is required in this task. An explicitly gated grouped-point implementation can realize the same calculation.",
        "Call the task a controlled address-demand test and preserve the distinction between information requirement and physical implementation.",
        "We next ask whether a small nested route basis can represent a richer hierarchy efficiently.",
    ),
    note(
        "0:45",
        "A hierarchical task tests subtree routes at limited feedback bandwidth",
        "Define hierarchy, feedback budget, and matched controls before presenting the narrow topology effect.",
        "Eight context streams occupy leaves of a known hierarchy. The target selects one stream, while hierarchy-dependent distractors create structured credit. K is the number of independent feedback coefficients: one, two, four, or eight. We compare correct ancestry routes with cyclic derangement, depth- and degree-preserving rewiring, and matched dense low-rank bases. These controls separate correct ownership of a signal from the smaller question of whether nested tree supports are an efficient basis at restricted bandwidth.",
        "Start with the task tree, then read the route dictionaries at each K. K counts backward coefficients; it does not count physical dendritic stages.",
        "Is the hierarchy discovered from the data?",
        "No. The hierarchy is deliberately built into the task so that the predicted route match can be tested under controlled conditions.",
        "State which resources are matched: rank, sparsity or feedback cost where applicable, and the same forward architecture.",
        "The result separates a large ownership effect from a much smaller topology-specific increment.",
    ),
    note(
        "0:45",
        "Correct assignment is essential; ancestry adds only a narrow K=4 gain",
        "Report the hierarchy result without conflating ownership and anatomy.",
        "At K equal to four, sending coordinates to their correct supports beats cyclic derangement by 61.1 percentage points. That is the ownership effect. Against the strongest matched non-anatomical low-rank control, ancestry routes add only 1.27 points, and only at this intermediate bandwidth. At lower bandwidth the basis is too coarse; at full bandwidth generic operators already span the field. Thus most of the benefit is correct signal assignment, while nested topology supplies a small efficiency increment in the theory’s predicted interior regime.",
        "Read the full K curve before focusing on the K equals four contrast. Keep the 61.1-point ownership comparison visually and verbally separate from the 1.27-point topology comparison.",
        "Why call the topology effect meaningful if it is only 1.27 points?",
        "Because it occurs at the predicted intermediate bandwidth and survives matched controls, but it is properly interpreted as a narrow efficiency effect rather than a broad anatomical advantage.",
        "Never headline the ownership contrast as a pure dendritic-topology benefit.",
        "Backward route resolution is only one role for dendrites; the next experiment tests forward serial computation.",
    ),
    note(
        "0:45",
        "Physical depth asks whether serial shunting matches a compositional task",
        "Define the multiplicative nuisance task, the depth notation, and all forward-model controls.",
        "The observed excitatory stream is a class signal multiplied by nested nuisance gains across H_p latent levels. A serial dendritic model has D_p ordered shunting stages whose inhibitory sensors can cancel those gains. Sensor alignment alpha ranges from independent sensors at zero to exact latent-gain sensors at one; these are forward inhibitory inputs, not learning signals. We compare nested factors, multiplicative factors distributed across non-nested groups, and local-ratio tasks. Controls include a resource-identical parallel grouped-point model, a raw-additive tree, and a flexible parameter-matched point-network ceiling. D_p is physical forward depth, not feedback bandwidth K.",
        "Trace one product of nuisance factors into the serial stages and show how the aligned inhibitory sensors divide them out in order. Then point to the grouped and flexible point controls.",
        "Was this task selected because it favors shunting?",
        "Yes, deliberately. It is an existence test of the hypothesis that ordered divisive stages are a task-matched inductive bias, with controls that bound how specific that advantage is.",
        "Always disclose the compositional generator and the calibrated operating regime alongside the effect.",
        "The result asks whether operation, order, task, and local feedback all have to match.",
    ),
    note(
        "0:45",
        "Serial depth helps only when operation, order, and task are aligned",
        "Give the paper’s strongest positive depth result together with its boundaries.",
        "On the aligned compositional task, D3 exceeds D1 by 30.9 percentage points under backpropagation. At D3, exact compartment-resolved local feedback recovers about 11 points over repeating one somatic signal. Reversing or independently shuffling sensors removes the depth benefit, and raw-additive depth is harmful. A resource-identical parallel grouped-point model does not reproduce the serial effect, while a flexible point network exceeds the tree and useful depth saturates. Serial dendrites therefore provide a constrained matched inductive bias, not a universal expressivity advantage.",
        "Read the main D1-to-D3 contrast, then use the alignment curve and controls to show that the result depends on divisive operation, serial order, and task structure.",
        "Does this show that deeper dendritic trees generally learn better?",
        "No. Fixed-budget depth is often costly. It helps here only because the ordered shunting computation matches a deliberately compositional task.",
        "Pair the 30.9-point headline with the matched-task disclosure, grouped-point control, flexible point ceiling, and saturation.",
        "We now move from controlled models to the route capacity of reconstructed cortical arbors.",
    ),
    note(
        "0:50",
        "Reconstructed arbors provide sparse, predominantly coarse route dictionaries",
        "Test whether real morphology economically spans modeled credit fields without claiming endogenous use.",
        "On reconstructed arbors, nested subtree routes define a sparse dictionary. Capture is the fraction of a modeled credit field’s energy contained in that route span. Eight channels retain about 85 percent of the dense ceiling with roughly 6.9 percent of dense feedback connections. The model-matched ceiling is 14.2-fold capture per connection, while the more conservative density-matched anatomy-specific advantage is about 2.7-fold. A disjoint 47-cell cohort reproduces the route ordering, and all ten qualifying cells from a second mouse show the model-matched subtree advantage. Much of the capacity is nevertheless explained by coarse branching and depth.",
        "Begin with the reconstruction and route supports, then read capture against wiring. Treat the 14.2-fold and 2.7-fold values as different denominators, not interchangeable summaries.",
        "Are these measured learning signals on the reconstructed neurons?",
        "No. These are modeled credit fields evaluated on measured anatomy. They establish representational capacity, not biological use during learning.",
        "Always pair the model-matched ceiling with the density-matched anatomical control and state the cohort scope.",
        "Morphology supplies routes; the next question is whether conductance can regulate their gain selectively.",
    ),
    note(
        "0:55",
        "Focal shunting regulates route gain only in a permissive conductance regime",
        "Present the focal-inhibition mechanism together with its standard-passive null.",
        "The derivative of log route gain with respect to an inhibitory conductance is negative only for compartments whose ancestry path includes the perturbed site. This predicts descendant-selective modulation. We compare a focal shunt with a baseline-current-matched additive injection while restoring somatic voltage in both cases. Under the standard passive calibration, R_m equal to 15,000 ohm square centimeters, the effect is null. Descendant localization appears only at lower membrane resistance or with background conductance—a permissive high-conductance regime.",
        "Use the ancestry indicator in the equation to identify which descendants can change. Then compare the standard-passive point with the permissive dose-response regime.",
        "Does this show that inhibitory synapses carry error signals in vivo?",
        "No. It shows a state-dependent biophysical mechanism by which focal conductance can regulate an already addressed transport field.",
        "State the standard-calibration null at first mention; do not present the permissive regime as generic.",
        "Finally, we ask whether measured functional responses preferentially align with these anatomical routes.",
    ),
    note(
        "0:55",
        "Measured visual responses do not preferentially align with ancestry routes",
        "State the main functional null cleanly and preserve the inferential scope.",
        "Measured presynaptic visual responses are mapped to dendritic compartments on reconstructed MICrONS target cells. Route dictionaries then predict each target cell’s held-out visual response. For seven target cells from one mouse, complete-tree ancestry routes do not outperform matched controls: topology-minus-control effects cross zero. This is an informative boundary. The anatomy provides candidate addresses, but the measured prediction task does not show that visual response structure preferentially occupies their span or that cortex uses them for local credit assignment.",
        "Read the left pipeline as measured response to route-based held-out prediction, then inspect the paired topology-minus-control effects. Values on both sides of zero indicate no consistent morphology-specific advantage.",
        "Does a null in seven cells rule out dendritic credit routing?",
        "No. It rules out a detectable advantage for these route dictionaries in this measured task and cohort. It does not establish what happens during learning or in other animals and tasks.",
        "Say seven target cells and one mouse explicitly; avoid treating candidate-route availability as endogenous use.",
        "A controlled rotation then asks whether the same fixed anatomy would help if the required credit field were aligned to it.",
    ),
    note(
        "0:55",
        "The same anatomy becomes useful when credit is rotated into its route span",
        "Demonstrate conditional representational sufficiency without confusing imposed and endogenous alignment.",
        "We decompose a fixed-energy credit field into components parallel and orthogonal to the anatomical route span, then vary alignment a without changing total energy. At a equal to zero, anatomy loses to matched controls in all eight cases. At a equal to one, it wins in all eight, and capture correlates with one-step progress at rho 0.99. Thus the anatomy can be useful when the task-credit geometry is aligned to it. The alignment is imposed experimentally; the result does not show that cortical learning creates it.",
        "Follow the field rotation from orthogonal to parallel and then the paired progress changes. The construction isolates alignment while holding energy fixed.",
        "Is this a trained biological-alignment result?",
        "No. It is a controlled sufficiency test on fixed anatomy and modeled fields, not evidence that alignment is learned or observed endogenously.",
        "Use the phrase imposed-alignment rescue and keep it distinct from the measured-response null.",
        "All positive and negative results can now be placed on one alignment-by-bandwidth map.",
    ),
    note(
        "0:50",
        "Dendritic benefit requires a match among coordinate, address, gain, and task",
        "Close with the unified conditional answer and return to the four objectives.",
        "The horizontal axis is task-route alignment; the vertical axis is feedback bandwidth relative to effective credit rank. At low bandwidth, the signal lacks the coordinates needed to distinguish neurons or branches. At weak alignment, anatomical routes discard useful credit. At full bandwidth, generic operators already span the field and anatomy loses a unique advantage. Dendritic structure is most useful in the intermediate matched regime, with sufficient alignment and reliable transport. The map is theoretical, not fitted. The result is a boundary theory: neuron coordinates solve standard tasks, branch addresses solve conflict, serial shunting helps matched composition, and endogenous morphology-specific alignment remains unestablished.",
        "Place each controlled positive and null on the phase plane, then return to coordinate, address, gain, and alignment as the four conditions that jointly determine usefulness.",
        "What is the strongest testable prediction?",
        "Branch-resolved plasticity should grow with task-route alignment at intermediate resolution, while focal inhibition should modulate it mainly in high-conductance states.",
        "The phase plane is a theoretical synthesis, not a fitted claim about where natural cortical learning operates.",
        "End with the conditional statement: dendrites can route and transform local credit, but they do not generically replace backpropagation.",
    ),
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
.script p { font-size: 14px; line-height: 1.38; }
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
            "Scripted time: 18:10. The remaining 1:50 is reserved for pauses, emphasis, and transitions.",
        )
        builder.MARKDOWN.write_text(text, encoding="utf-8")

    def build_html() -> None:
        original_html()
        text = builder.HTML_OUT.read_text(encoding="utf-8")
        text = text.replace("Kempner Learning Dynamics Workshop", "Dendrites and Credit Assignment Workshop")
        builder.HTML_OUT.write_text(text, encoding="utf-8")

    def validate() -> None:
        import fitz

        if len(builder.NOTES) != 25:
            raise SystemExit(f"Expected 25 notes, found {len(builder.NOTES)}")
        with fitz.open(builder.PDF_OUT) as document:
            if document.page_count != len(builder.NOTES):
                raise SystemExit(f"Speaker guide has {document.page_count} pages")
            for index, page in enumerate(document, start=1):
                if page.rect.width <= page.rect.height:
                    raise SystemExit(f"Guide page {index} is not landscape")
                if len(page.get_text().strip()) < 500:
                    raise SystemExit(f"Guide page {index} has unexpectedly little text")
        print("Validated 25 speaker-guide pages; scripted time 18:10")

    builder.build_markdown = build_markdown
    builder.build_html = build_html
    builder.validate = validate
    builder.main()


if __name__ == "__main__":
    main()
