# Frozen optimization-sensitivity plan

This plan was written before inspecting the 1,200-step learning-rate sweep.
It addresses the possibility that the original 600-step analysis stopped before
the training objectives stabilized.

## Fixed design

- Learning rates: 0.01, 0.02 and 0.04.
- Training length: 1,200 full-batch Adam steps.
- All seven target cells and all ten held-out-stimulus splits are retained.
- The base seed, stimulus splits, initial states, selected morphology routes,
  site shuffles and random-route draws are identical across learning rates.
- No route or learning-rate setting will be chosen from held-out MSE.

## Prospective stability rule

For each run and method, define the late objective change as the final recorded
training objective minus the value 100 steps earlier. A setting is considered
numerically admissible only if all outputs are finite, no transformed
conductance parameter reaches its explicit numerical bound, and at least 90%
of runs finish below their initial training objective.

Among admissible settings, choose the learning rate with the smallest
worst-method median absolute late change divided by the absolute total
objective change. If this diagnostic is practically tied (within 25%), prefer
the smaller learning rate. Held-out accuracy, the topology-control ordering and
statistical significance are not selection criteria.

The originally frozen 0.02/600 setting remains an audit reference. It will be
replaced as the publication setting only if the prospective rule selects a
1,200-step condition and the qualitative scientific boundary is stable.
