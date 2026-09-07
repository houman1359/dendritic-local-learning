# Independent completed MNIST numeric audit

All means and confidence intervals reproduce. The audit reconstructed all 298 endpoint rows from canonical per-run records and recomputed all 12 rate selections from development validation loss alone. It verified all 180 epochs, pairing of initialized models, unchanged decoder-only cores and all 40 capture checkpoint identities.

| file | n_summary_rows | maximum_absolute_difference | status |
| --- | --- | --- | --- |
| condition_summary_six_rules.csv | 96 | 1.1102230246251565e-16 | means_and_intervals_passed |
| paired_contrasts_six_rules.csv | 56 | 9.93129189996722e-17 | means_and_intervals_passed |
| development_rate_summary_six_rules.csv | 144 | 1.4210854715202004e-14 | means_and_intervals_passed |
| delivery_coordinate_capture_summary.csv | 48 | 1.1102230246251565e-16 | means_and_intervals_passed |

Headline accuracy differences are in percentage points:

| rate_policy | architecture | metric | contrast | mean | ci_low | ci_high | n | positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| common_original | additive | test_accuracy | neuron_shared_minus_strict_scalar | 7.840999960899353 | 7.22499905526638 | 8.594999924302101 | 10 | 10 |
| common_original | additive | test_accuracy | subtree_k3_minus_projected_k1 | -0.012999773025512695 | -0.0689995437860489 | 0.047002434730529785 | 10 | 3 |
| common_original | additive | test_accuracy | exact_path_minus_subtree_k3 | 0.06500005722045898 | 0.02300022542476654 | 0.11300027370452881 | 10 | 7 |
| common_original | additive | test_accuracy | neuron_shared_minus_decoder_only | 7.536999583244324 | 7.314000725746155 | 7.776998281478882 | 10 | 10 |
| common_original | additive | test_accuracy | exact_path_minus_neuron_shared | 0.1910001039505005 | 0.09400069713592529 | 0.2879995107650757 | 10 | 8 |
| common_original | shunting | test_accuracy | neuron_shared_minus_strict_scalar | 11.286999583244324 | 10.499998912215233 | 11.962999701499939 | 10 | 10 |
| common_original | shunting | test_accuracy | subtree_k3_minus_projected_k1 | 0.006000399589538574 | -0.03500103950500488 | 0.04600226879119873 | 10 | 5 |
| common_original | shunting | test_accuracy | exact_path_minus_subtree_k3 | -0.04999995231628418 | -0.11299967765808105 | 0.015001296997070314 | 10 | 4 |
| common_original | shunting | test_accuracy | neuron_shared_minus_decoder_only | 21.19700014591217 | 20.733999609947205 | 21.62599864602089 | 10 | 10 |
| common_original | shunting | test_accuracy | exact_path_minus_neuron_shared | 0.010000467300415039 | -0.061997175216674805 | 0.07499933242797852 | 10 | 5 |
| selected | additive | test_accuracy | neuron_shared_minus_strict_scalar | 7.45599925518036 | 6.912999153137207 | 8.086998462677002 | 10 | 10 |
| selected | additive | test_accuracy | subtree_k3_minus_projected_k1 | 0.036000609397888184 | -0.05599856376647949 | 0.12200057506561279 | 10 | 5 |
| selected | additive | test_accuracy | exact_path_minus_subtree_k3 | -0.04400014877319336 | -0.14300107955932617 | 0.0580000877380371 | 10 | 4 |
| selected | additive | test_accuracy | neuron_shared_minus_decoder_only | 7.545999288558961 | 7.32800006866455 | 7.757997527718544 | 10 | 10 |
| selected | additive | test_accuracy | exact_path_minus_neuron_shared | 0.18200039863586426 | 0.11400043964385985 | 0.252000093460083 | 10 | 10 |
| selected | shunting | test_accuracy | neuron_shared_minus_strict_scalar | 9.22999918460846 | 8.593000173568726 | 9.82999862730503 | 10 | 10 |
| selected | shunting | test_accuracy | subtree_k3_minus_projected_k1 | 0.04900038242340088 | -0.04802575707435594 | 0.1410001516342163 | 10 | 7 |
| selected | shunting | test_accuracy | exact_path_minus_subtree_k3 | -0.09299993515014648 | -0.1949995756149292 | 0.009000897407531738 | 10 | 3 |
| selected | shunting | test_accuracy | neuron_shared_minus_decoder_only | 16.167999505996704 | 15.612999796867372 | 16.658998742699623 | 10 | 10 |
| selected | shunting | test_accuracy | exact_path_minus_neuron_shared | 0.010000467300415039 | -0.061997175216674805 | 0.07499933242797852 | 10 | 5 |

The audit imported no study aggregators and changed no scientific files. Confidence intervals use 50,000 resamples of paired training seeds with random seed 911. These are finite-budget comparisons; intervals spanning zero are not evidence of practical equivalence. K1 and K3 preserve the same per-example-neuron common-mode mean at identical parameters, rather than matching total vector norm.
