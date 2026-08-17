# Fixed spatial-topology audit

The spatial and random networks contain the same number of trainable contacts,
but they do not have the same input coverage. Across 10 topology seeds
and 128 neurons per seed, the random map covers 276.2 unique
pixels per neuron on average, whereas the spatial map covers exactly
336. Mean pairwise branch overlap is 0.0139 for
random maps and 0.0000 for spatial maps. This follows from the
registered spatial sampler: its 16 distal branch regions are disjoint and each
contains 21 unique contacts.

The spatial-minus-random accuracy effect is also present under matched
backpropagation (mean 0.72 percentage points across the input-valid
task--core conditions). MNIST averages additive and shunting cores, whereas the
randomly projected noise task uses the additive core only: the historical
signed-input positive-conductance shunting cells are excluded independently of
outcome. The retained experiment therefore demonstrates a forward sparse-
connectivity prior, but it does not isolate task-aligned dendritic credit
routing. We retain it as a boundary/control result and do not use it as
evidence that spatial topology improves local credit assignment specifically.
