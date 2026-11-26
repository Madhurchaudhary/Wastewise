# Model comparison

Selection rule: highest validation macro-F1; ties within 0.5 pp broken by lower CPU latency.
Test metrics are reported for information only and were not used for selection.

| experiment | architecture | params_m | size_mb | epochs | train_minutes | val_acc | val_f1_macro | test_acc | test_precision_macro | test_recall_macro | test_f1_macro | test_f1_weighted | test_roc_auc | hazardous_recall | e_waste_recall | threshold | coverage | acc_on_accepted | ood_rejection | latency_ms_cpu | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp01_custom_cnn | custom_cnn | 1.96 | 7.86 | 12 | 46.6 | 0.5256 | 0.4951 | 0.5527 | 0.5342 | 0.5362 | 0.5184 | 0.5508 | 0.8555 | 0.3953 | 0.7665 | 0.87 | 0.0393 | 0.9268 | 0.9917 | 20.3 | False |
| exp02_mobilenet_v3_large | mobilenet_v3_large | 3.48 | 14.13 | 15 | 47.6 | 0.9151 | 0.9049 | 0.9234 | 0.9121 | 0.929 | 0.9192 | 0.9234 | 0.9935 | 0.9767 | 0.982 | 0.59 | 0.9464 | 0.9464 | 0.3567 | 18.1 | False |
| exp03_efficientnet_b0 | efficientnet_b0 | 4.68 | 19.02 | 10 | 52.3 | 0.9142 | 0.9078 | 0.9071 | 0.8911 | 0.9146 | 0.9 | 0.9071 | 0.9904 | 0.9767 | 0.976 | 0.59 | 0.9377 | 0.9336 | 0.3367 | 30.1 | False |
| exp04_mobilenet_v3_large_v11 | mobilenet_v3_large | 3.48 | 14.13 | 6 | 30.5 | 0.9354 | 0.9252 | 0.9262 | 0.9163 | 0.929 | 0.9216 | 0.9263 | 0.9943 | 0.9535 | 0.988 | 0.65 | 0.9444 | 0.9564 | 0.3783 | 14.6 | True |
