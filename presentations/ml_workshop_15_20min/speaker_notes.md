# Speaker notes — ML workshop deck

The durations below total approximately 18.5 minutes. Each slide begins with
the sentence to say before discussing its equation or plot.

## 1. When dendritic structure helps local credit assignment — 0:20

**Opening intuition:** Dendrites may help learning not by replacing
backpropagation, but by structuring where a learning signal acts inside one
neuron.

Today I will separate three resources: a signal that identifies the neuron, an
address that selects a subtree, and a conductance-dependent gain along that
route. The main result is conditional: these resources help only when they
match distinctions required by the task.

## 2. Credit assignment is a routing problem — 0:45

**Opening intuition:** A single behavioral outcome must be converted into many
local parameter changes.

Define parameter credit as the derivative of the loss with respect to one
parameter. The spatial problem becomes progressively sharper from network to
neuron to branch to synapse. Plasticity can be local even though the
task-dependent information was computed elsewhere.

## 3. Backpropagation provides exact parameter credit — 0:55

**Before the equation:** Backpropagation is our information reference, not our
assumed biological mechanism.

The presynaptic activation is local. The reverse variable is
parameter-specific and recursively combines downstream Jacobians. Exact
backpropagation supplies a destination, sign and magnitude for every weight.
Biological learning theories can replace different parts of this computation,
but they must still supply enough task-dependent information.

## 4. A local rule preserves eligibility and approximates the error — 0:55

**Before the equation:** Local learning preserves the local factor and changes
how the task-dependent factor is obtained.

The eligibility can use presynaptic activity, membrane voltage, driving force
and local gates. The communicated learning signal may still depend on a
global objective. Therefore “local” does not mean “uninformed”; feedback
bandwidth determines which distinctions the rule can express.

## 5. The first bottleneck is which neuron should learn — 0:45

**Opening intuition:** Before asking which branch should learn, feedback must
identify the correct cell.

A single scalar does not collapse neurons into copies: every synapse retains a
different eligibility and initialization. It does remove task-specific
coordinates from the feedback. One signal per neuron restores neuron identity;
multiple signals inside one tree add a separate within-neuron address.

## 6. A dendritic arbor adds state, addresses and route gain — 0:45

**Opening intuition:** Dendrites add three conceptually different resources.

Local state changes the synaptic Jacobian. Subtree supports form a structured
low-rank basis for spatial credit. Conductance can scale error transport along
a route. The scientific question is whether these resources align with the
task—not simply whether a dendritic model has more parameters.

## 7. Conductance makes voltage a normalized quotient — 0:55

**Before the equation:** Conductance affects both the numerator that drives
voltage and the denominator that sets sensitivity.

The inverse total conductance is local input resistance. An additive current
can match a voltage perturbation without changing this denominator; a shunt
changes both. This is why a shunt can modify the gains of other synapses even
when its own net current is small.

## 8. Eligibility stays local; the error becomes a field — 1:10

**Before the equation:** The exact dendritic gradient has the same logical
form as a point-neuron three-factor rule.

The first factor uses presynaptic activity, local input resistance and driving
force. The second is the loss derivative assigned to that compartment. In the
directed tree it is the somatic error multiplied by a path product; in a
general compartmental model it is the steady-state adjoint. Dendrites do not
generate the circuit-level error. They provide a spatial transformation after
that error reaches the neuron.

## 9. We test progressively richer credit — 0:30

**Opening intuition:** The experiments move from ordinary tasks to controlled
necessity tests and then to biology.

Each positive result has a point or grouped-point implementation receiving the
same routed field. Those controls distinguish the informational value of an
address from claims that dendritic material is uniquely required.

## 10. Neuron-specific feedback supplies nearly all useful resolution — 0:55

**Opening intuition:** Harder image classification does not automatically make
within-tree routing useful.

On MNIST, strict scalar to neuron-specific feedback adds 11.25 percentage
points in the shunting model and 7.81 points in the additive model. Exact path
resolution then adds only 0.045 and 0.186 points. On flattened CIFAR-10,
neuron-specific feedback adds 16.39 points over scalar, while exact path is
0.86 points below neuron-specific. This is a boundary result, not a claim that
compartment feedback can never help.

## 11. Restricted feedback helps only when it rejects more noise than signal — 1:10

**Before the equation:** A restricted pathway filters both useful gradient
signal and stochastic noise.

The credit operator `M` selects, assigns and scales the stochastic gradient.
Its utility increases with aligned signal and decreases with update cost and
admitted noise. Across 540 conditions, utility correlates with observed
one-step progress at ρ=0.937 and final accuracy at ρ=0.916. The bound is exact
for a quadratic loss and otherwise a guaranteed one-step statement; it does
not predict every difference accumulated during nonlinear training.

**Transition:** The theory predicts a sharp need for an address when shared
feedback mixes incompatible local updates.

## 12. Credit conflict creates a demand for branch-specific signals — 1:00

**Opening intuition:** We construct a task in which all branches are active,
but only one branch should determine the label.

Each branch receives a Fashion-MNIST image. Context selects the forward
branch. Conflict probability χ changes nonselected branches from the same
class to the opposite class. At χ=0, one shared signal is harmless. At high χ,
shared feedback applies the selected branch's error to eligibilities that call
for the opposite update.

## 13. Shared credit fails at the predicted conflict boundary — 0:55

**Before the boundary:** The shared update loses its descent component at a
branch-count-dependent conflict dose.

The predicted thresholds are 1, 2/3 and 4/7 for 2, 4 and 8 branches. Trained
collapse points follow these values. At full conflict, correct routing gains
34.9, 58.2 and 57.4 percentage points over shared feedback. A cyclically
deranged route fails, while analytic backpropagation and a context-gated point
model match correct routing. This proves a requirement for an address, not a
unique requirement for dendrites.

## 14. Nested tasks ask whether a few subtree addresses are efficient — 0:55

**Opening intuition:** The next task asks whether a tree is a useful compressed
basis when feedback bandwidth is limited.

Eight active streams sit at the leaves of a known hierarchy. `K` is the number
of independently communicated signals within one neuron. The controls match
rank, sparsity, parameter count and forward resources while changing route
assignment, basis or topology.

## 15. Subtree addresses help, but topology adds only a narrow gain — 0:55

**Opening intuition:** The value of having addresses is large; the additional
value of this particular tree basis is much smaller.

At `K=4`, matched subtrees beat one neuron-shared signal by about 61 points.
Against the strongest matched non-anatomical low-rank control, the gain is only
1.27 points. Matched subtrees lose at low bandwidth and tie at full rank;
rewiring removes the intermediate gain. Identical routed fields produce
identical dendritic and point-model behavior.

## 16. Physical depth helps only when serial computation matches the task — 1:20

**Opening intuition:** Forward dendritic depth is distinct from backward route
resolution.

Here `D_p` counts serial physical stages, `H` is task hierarchy and `α` is
task–sensor alignment. On the deliberately calibrated `H=3` task, aligned D3
beats D1 by 30.9 points under backpropagation. Independent, shuffled and
reversed sensors remove the effect. At D3, exact compartment feedback adds 11
points over one shared somatic signal. Useful depth saturates, can hurt when
ratios are already local, and is exceeded by a flexible parameter-matched
point MLP. Present this as a matched-composition existence proof.

## 17. Real arbors provide sparse candidate routes — 0:55

**Opening intuition:** The controlled tasks establish when an address could
help; anatomy asks whether such addresses exist.

Branch points define nested supports. At eight channels, subtree routes retain
85% of dense field capture with about 7% of the dense feedback connections.
The cellwise wiring-normalized advantage over a same-density shuffled
dictionary is approximately 2.8-fold. This ordering appears in a disjoint
47-cell cohort and ten quality-controlled cells from a second mouse. These are
modeled fields on measured anatomy, not observed learning signals.

## 18. Focal shunting changes descendant credit only conditionally — 1:00

**Opening intuition:** A focal shunt can alter the feedback operator even when
an additive current matches its local voltage effect.

The Green's-function column determines the spatial spread. At the standard
passive calibration, the shunt-minus-current localization contrast has no
material effect. It emerges when membrane resistance is lowered or background
conductance is added and persists in steady-state active-channel extensions.
This is a state-dependent route-gain mechanism, not evidence that inhibition
generally improves learning.

## 19. Measured responses are null; imposed alignment rescues the routes — 1:10

**Opening intuition:** Anatomical availability and conductance-dependent gain
still do not establish that biological activity uses the routes.

In seven MICrONS target cells, nested subtrees do not beat random or
site-shuffled routes for held-out visual-response prediction or field capture.
When we hold anatomy and field energy fixed and rotate the task field into the
subtree span, the same routes change from losing at zero alignment to winning
at complete alignment. Capture and progress correlate at approximately 0.99.
This proves conditional sufficiency, not endogenous alignment.

## 20. Dendrites provide conditional resources for local credit — 0:50

**Closing answer:** Coordinates select neurons; subtree routes address
synapses; conductance changes route gain; alignment determines whether those
resources help.

One coordinate per neuron carries most of the practical benefit on ordinary
tasks. Within-neuron addresses matter when branches require different
task-dependent updates. Shunting can regulate route gain only in an appropriate
electrotonic regime. Measured morphology-specific use remains unestablished.

**Final sentence:** Dendritic structure is a conditional substrate for routing
local credit, not a general replacement for backpropagation.
