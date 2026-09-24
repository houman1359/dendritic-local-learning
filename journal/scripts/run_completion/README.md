# Completion of image and physical-depth controls

`analyze_controls.py --output /tmp/control-statistics` checks all 220 compact run records and reproduces the paired estimates, 50,000-draw bootstrap intervals and exact sign-flip tests. Fashion-MNIST primary contrasts and physical-placement interactions form separate two-test Holm families. All fits and seeds are retained.

Fashion-MNIST uses 80 fresh fits at the inherited fixed 180-epoch budget. The physical placement study uses 140 same-seed, recipe-matched sensitivity fits. Reaching a prespecified finite budget in these controls is not an execution failure or a claim of convergence.

`production/` preserves the exact production audit, analysis and repair drivers, including their historical cluster paths. These drivers document the execution; the portable statistical reanalysis above needs only the distributed Source Data. The storage repair changes output locations only for the placement controls. Physical stopping extensions preserve model, optimizer, random and data-loader states at continuation.

The CIFAR stopping extension reuses the original seeds; its eight reruns reproduce their observed loss prefixes and retain identical validation-selected test accuracies. Its scope must not be described as fresh confirmation.

For sanitized released Source Data, restore it with the release helper first; the analysis verifies each original-to-released hash link through `RELEASED_SOURCE_HASHES.tsv`. Numeric outcomes and original evidence hashes remain unchanged.

`analyze_stopping.py --output /tmp/stopping-statistics` reconstructs validation-only checkpoint selection from every observed epoch of the sixty physical-depth fits, verifies ordinary stopping, and reproduces the trajectory means, paired differences and bootstrap intervals. Selected metric records retain hashes of the corresponding raw evaluation files. The production checkpoint audit additionally verifies weights, optimizer/random states and configuration identity; those large raw checkpoints are retained with the execution archive rather than duplicated in the article bundle.

Figure 7 keeps its original 180-epoch architectural comparisons in A–D and uses the complete, uniform-H200 stopping extension in E–H. The original mixed-device 600-epoch study remains historical evidence. Both extensions reuse the original seeds and therefore do not constitute fresh independent confirmation.
