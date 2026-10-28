# Results summary (auto-generated)

alpha=0.1, score=lac, precisions present: ['ws_fp32']

Columns (test bearings, mean over folds): bacc_src = balanced acc. at source condition; bacc_shift / ece_shift = mean over shifted conditions (ece uncalibrated, ece_shift_ts after temperature scaling); agree = label agreement with dense FP32 on shifted conditions; cov_src / cov_mon = worst-condition coverage of source-calibrated / operating-point (Mondrian) conformal sets (target 0.90); size_mon = mean Mondrian set size.


## resnet1d

|                   |   bacc_src |   bacc_shift |   ece_shift |   ece_shift_ts |   agree |   cov_src |   cov_mon |   size_mon |
|:------------------|-----------:|-------------:|------------:|---------------:|--------:|----------:|----------:|-----------:|
| (0.0, 'ws_fp32')  |      0.507 |        0.434 |       0.475 |          0.201 |   1     |     0.833 |     0.746 |      2.603 |
| (0.3, 'ws_fp32')  |      0.405 |        0.399 |       0.496 |          0.178 |   0.865 |     0.707 |     0.698 |      2.545 |
| (0.5, 'ws_fp32')  |      0.437 |        0.403 |       0.468 |          0.193 |   0.908 |     0.801 |     0.791 |      2.75  |
| (0.7, 'ws_fp32')  |      0.487 |        0.417 |       0.389 |          0.19  |   0.82  |     0.877 |     0.805 |      2.818 |
| (0.85, 'ws_fp32') |      0.555 |        0.446 |       0.352 |          0.256 |   0.843 |     0.869 |     0.679 |      2.403 |

## wdcnn

|                   |   bacc_src |   bacc_shift |   ece_shift |   ece_shift_ts |   agree |   cov_src |   cov_mon |   size_mon |
|:------------------|-----------:|-------------:|------------:|---------------:|--------:|----------:|----------:|-----------:|
| (0.0, 'ws_fp32')  |      0.407 |        0.379 |       0.456 |          0.185 |   1     |     0.903 |     0.587 |      2.592 |
| (0.3, 'ws_fp32')  |      0.417 |        0.395 |       0.427 |          0.153 |   0.908 |     0.861 |     0.555 |      2.52  |
| (0.5, 'ws_fp32')  |      0.415 |        0.388 |       0.392 |          0.186 |   0.9   |     0.863 |     0.588 |      2.51  |
| (0.7, 'ws_fp32')  |      0.485 |        0.427 |       0.299 |          0.209 |   0.873 |     0.872 |     0.575 |      2.412 |
| (0.85, 'ws_fp32') |      0.521 |        0.455 |       0.195 |          0.189 |   0.822 |     0.757 |     0.654 |      2.302 |

## Deployment rule (mean over folds)

|                                           |   ratio |   cost |   bacc_shift |   ece_shift_ts |   cov_worst_src |   cov_worst_mondrian |   set_size_mondrian |
|:------------------------------------------|--------:|-------:|-------------:|---------------:|----------------:|---------------------:|--------------------:|
| ('resnet1d', 'accuracy-only')             |    0.85 |  0.023 |        0.446 |          0.256 |           0.869 |                0.679 |               2.403 |
| ('resnet1d', 'trust-aware (mondrian CP)') |    0    |  1     |        0.434 |          0.201 |           0.833 |                0.746 |               2.603 |
| ('resnet1d', 'trust-aware (src CP)')      |    0    |  1     |        0.434 |          0.201 |           0.833 |                0.746 |               2.603 |
| ('wdcnn', 'accuracy-only')                |    0.85 |  0.001 |        0.455 |          0.189 |           0.757 |                0.654 |               2.302 |
| ('wdcnn', 'trust-aware (mondrian CP)')    |    0    |  0.023 |        0.379 |          0.185 |           0.903 |                0.587 |               2.592 |
| ('wdcnn', 'trust-aware (src CP)')         |    0    |  0.023 |        0.379 |          0.185 |           0.903 |                0.587 |               2.592 |


Per-fold choices:

| fold   | arch     | rule                      |   ratio | precision   |   cost |   bacc_src |   bacc_shift |   ece_shift_raw |   ece_shift_ts |   cov_worst_src |   set_size_src |   cov_worst_mondrian |   set_size_mondrian |   fallback_dense |
|:-------|:---------|:--------------------------|--------:|:------------|-------:|-----------:|-------------:|----------------:|---------------:|----------------:|---------------:|---------------------:|--------------------:|-----------------:|
| A      | resnet1d | accuracy-only             |    0.85 | ws_fp32     |  0.023 |      0.555 |        0.446 |           0.352 |          0.256 |           0.869 |          2.641 |                0.679 |               2.403 |              nan |
| A      | resnet1d | trust-aware (src CP)      |    0    | ws_fp32     |  1     |      0.507 |        0.434 |           0.475 |          0.201 |           0.833 |          2.844 |                0.746 |               2.603 |                1 |
| A      | resnet1d | trust-aware (mondrian CP) |    0    | ws_fp32     |  1     |      0.507 |        0.434 |           0.475 |          0.201 |           0.833 |          2.844 |                0.746 |               2.603 |                1 |
| A      | wdcnn    | accuracy-only             |    0.85 | ws_fp32     |  0.001 |      0.521 |        0.455 |           0.195 |          0.189 |           0.757 |          2.434 |                0.654 |               2.302 |              nan |
| A      | wdcnn    | trust-aware (src CP)      |    0    | ws_fp32     |  0.023 |      0.407 |        0.379 |           0.456 |          0.185 |           0.903 |          2.933 |                0.587 |               2.592 |                1 |
| A      | wdcnn    | trust-aware (mondrian CP) |    0    | ws_fp32     |  0.023 |      0.407 |        0.379 |           0.456 |          0.185 |           0.903 |          2.933 |                0.587 |               2.592 |                1 |
