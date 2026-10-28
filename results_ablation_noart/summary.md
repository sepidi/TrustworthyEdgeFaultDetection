# Results summary (auto-generated)

alpha=0.1, score=lac, precisions present: ['ws_fp32']

Columns (test bearings, mean over folds): bacc_src = balanced acc. at source condition; bacc_shift / ece_shift = mean over shifted conditions (ece uncalibrated, ece_shift_ts after temperature scaling); agree = label agreement with dense FP32 on shifted conditions; cov_src / cov_mon = worst-condition coverage of source-calibrated / operating-point (Mondrian) conformal sets (target 0.90); size_mon = mean Mondrian set size; cov_lobo / size_lobo = bearing-level operating-point conformal (calibrated on out-of-bearing scores, tcfd.lobo).


## resnet1d

|                   |   bacc_src |   bacc_shift |   ece_shift |   ece_shift_ts |   agree |   cov_src |   cov_mon |   size_mon |   cov_lobo |   size_lobo |
|:------------------|-----------:|-------------:|------------:|---------------:|--------:|----------:|----------:|-----------:|-----------:|------------:|
| (0.0, 'ws_fp32')  |      0.627 |        0.553 |       0.322 |          0.295 |   1     |     0.561 |     0.601 |      1.611 |      0.794 |       1.938 |
| (0.3, 'ws_fp32')  |      0.633 |        0.542 |       0.334 |          0.234 |   0.896 |     0.651 |     0.745 |      1.753 |      0.78  |       1.899 |
| (0.5, 'ws_fp32')  |      0.622 |        0.55  |       0.283 |          0.256 |   0.868 |     0.607 |     0.674 |      1.7   |      0.741 |       1.814 |
| (0.7, 'ws_fp32')  |      0.636 |        0.554 |       0.2   |          0.275 |   0.77  |     0.605 |     0.711 |      1.775 |      0.777 |       1.752 |
| (0.85, 'ws_fp32') |      0.636 |        0.574 |       0.152 |          0.212 |   0.664 |     0.678 |     0.709 |      1.618 |      0.837 |       1.873 |

## wdcnn

|                   |   bacc_src |   bacc_shift |   ece_shift |   ece_shift_ts |   agree |   cov_src |   cov_mon |   size_mon |   cov_lobo |   size_lobo |
|:------------------|-----------:|-------------:|------------:|---------------:|--------:|----------:|----------:|-----------:|-----------:|------------:|
| (0.0, 'ws_fp32')  |      0.657 |        0.56  |       0.22  |          0.324 |   1     |     0.473 |     0.7   |      1.763 |      0.766 |       1.867 |
| (0.3, 'ws_fp32')  |      0.61  |        0.526 |       0.253 |          0.3   |   0.755 |     0.479 |     0.684 |      1.781 |      0.716 |       1.859 |
| (0.5, 'ws_fp32')  |      0.603 |        0.517 |       0.189 |          0.244 |   0.715 |     0.584 |     0.791 |      1.958 |      0.823 |       1.948 |
| (0.7, 'ws_fp32')  |      0.632 |        0.551 |       0.111 |          0.227 |   0.622 |     0.683 |     0.77  |      2.033 |      0.752 |       1.986 |
| (0.85, 'ws_fp32') |      0.534 |        0.44  |       0.119 |          0.269 |   0.533 |     0.588 |     0.705 |      2.01  |      0.694 |       2.178 |

## Deployment rule (mean over folds)

|                                                |   ratio |   cost |   bacc_shift |   ece_shift_ts |   cov_worst_src |   cov_worst_mondrian |   set_size_mondrian |   cov_worst_lobo |   set_size_lobo |
|:-----------------------------------------------|--------:|-------:|-------------:|---------------:|----------------:|---------------------:|--------------------:|-----------------:|----------------:|
| ('resnet1d', 'accuracy-only')                  |     0.7 |  0.091 |        0.554 |          0.275 |           0.605 |                0.711 |               1.775 |            0.777 |           1.752 |
| ('resnet1d', 'trust-aware (lobo_mondrian CP)') |     0   |  1     |        0.553 |          0.295 |           0.561 |                0.601 |               1.611 |            0.794 |           1.938 |
| ('resnet1d', 'trust-aware (mondrian CP)')      |     0   |  1     |        0.553 |          0.295 |           0.561 |                0.601 |               1.611 |            0.794 |           1.938 |
| ('resnet1d', 'trust-aware (src CP)')           |     0   |  1     |        0.553 |          0.295 |           0.561 |                0.601 |               1.611 |            0.794 |           1.938 |
| ('wdcnn', 'accuracy-only')                     |     0.1 |  0.019 |        0.563 |          0.314 |           0.433 |                0.703 |               1.757 |            0.75  |           1.849 |
| ('wdcnn', 'trust-aware (lobo_mondrian CP)')    |     0   |  0.023 |        0.56  |          0.324 |           0.473 |                0.7   |               1.763 |            0.766 |           1.867 |
| ('wdcnn', 'trust-aware (mondrian CP)')         |     0   |  0.023 |        0.56  |          0.324 |           0.473 |                0.7   |               1.763 |            0.766 |           1.867 |
| ('wdcnn', 'trust-aware (src CP)')              |     0   |  0.023 |        0.56  |          0.324 |           0.473 |                0.7   |               1.763 |            0.766 |           1.867 |


Per-fold choices:

| fold   | arch     | rule                           |   ratio | precision   |   cost |   bacc_src |   bacc_shift |   ece_shift_raw |   ece_shift_ts |   cov_worst_src |   set_size_src |   cov_worst_mondrian |   set_size_mondrian |   cov_worst_lobo_mondrian |   set_size_lobo_mondrian |   fallback_dense |
|:-------|:---------|:-------------------------------|--------:|:------------|-------:|-----------:|-------------:|----------------:|---------------:|----------------:|---------------:|---------------------:|--------------------:|--------------------------:|-------------------------:|-----------------:|
| A      | resnet1d | accuracy-only                  |     0.7 | ws_fp32     |  0.091 |      0.514 |        0.473 |           0.276 |          0.23  |           0.485 |          1.727 |                0.792 |               1.919 |                     0.807 |                    2.032 |              nan |
| A      | resnet1d | trust-aware (src CP)           |     0   | ws_fp32     |  1     |      0.42  |        0.386 |           0.499 |          0.316 |           0.377 |          1.393 |                0.609 |               1.832 |                     0.685 |                    2.049 |                1 |
| A      | resnet1d | trust-aware (mondrian CP)      |     0   | ws_fp32     |  1     |      0.42  |        0.386 |           0.499 |          0.316 |           0.377 |          1.393 |                0.609 |               1.832 |                     0.685 |                    2.049 |                1 |
| A      | resnet1d | trust-aware (lobo_mondrian CP) |     0   | ws_fp32     |  1     |      0.42  |        0.386 |           0.499 |          0.316 |           0.377 |          1.393 |                0.609 |               1.832 |                     0.685 |                    2.049 |                1 |
| A      | wdcnn    | accuracy-only                  |     0   | ws_fp32     |  0.023 |      0.536 |        0.501 |           0.271 |          0.349 |           0.399 |          0.952 |                0.532 |               1.706 |                     0.782 |                    2.052 |              nan |
| A      | wdcnn    | trust-aware (src CP)           |     0   | ws_fp32     |  0.023 |      0.536 |        0.501 |           0.271 |          0.349 |           0.399 |          0.952 |                0.532 |               1.706 |                     0.782 |                    2.052 |                1 |
| A      | wdcnn    | trust-aware (mondrian CP)      |     0   | ws_fp32     |  0.023 |      0.536 |        0.501 |           0.271 |          0.349 |           0.399 |          0.952 |                0.532 |               1.706 |                     0.782 |                    2.052 |                1 |
| A      | wdcnn    | trust-aware (lobo_mondrian CP) |     0   | ws_fp32     |  0.023 |      0.536 |        0.501 |           0.271 |          0.349 |           0.399 |          0.952 |                0.532 |               1.706 |                     0.782 |                    2.052 |                1 |
| B      | resnet1d | accuracy-only                  |     0.7 | ws_fp32     |  0.091 |      0.759 |        0.625 |           0.166 |          0.203 |           0.972 |          2.296 |                0.868 |               2.154 |                     0.831 |                    1.675 |              nan |
| B      | resnet1d | trust-aware (src CP)           |     0   | ws_fp32     |  1     |      0.82  |        0.697 |           0.176 |          0.223 |           0.986 |          2.15  |                0.756 |               1.871 |                     0.877 |                    1.774 |                1 |
| B      | resnet1d | trust-aware (mondrian CP)      |     0   | ws_fp32     |  1     |      0.82  |        0.697 |           0.176 |          0.223 |           0.986 |          2.15  |                0.756 |               1.871 |                     0.877 |                    1.774 |                1 |
| B      | resnet1d | trust-aware (lobo_mondrian CP) |     0   | ws_fp32     |  1     |      0.82  |        0.697 |           0.176 |          0.223 |           0.986 |          2.15  |                0.756 |               1.871 |                     0.877 |                    1.774 |                1 |
| B      | wdcnn    | accuracy-only                  |     0.3 | ws_fp32     |  0.012 |      0.767 |        0.674 |           0.156 |          0.161 |           0.792 |          2.055 |                0.907 |               2.14  |                     0.674 |                    1.558 |              nan |
| B      | wdcnn    | trust-aware (src CP)           |     0   | ws_fp32     |  0.023 |      0.795 |        0.667 |           0.168 |          0.191 |           0.912 |          2.177 |                0.899 |               2.159 |                     0.721 |                    1.61  |                1 |
| B      | wdcnn    | trust-aware (mondrian CP)      |     0   | ws_fp32     |  0.023 |      0.795 |        0.667 |           0.168 |          0.191 |           0.912 |          2.177 |                0.899 |               2.159 |                     0.721 |                    1.61  |                1 |
| B      | wdcnn    | trust-aware (lobo_mondrian CP) |     0   | ws_fp32     |  0.023 |      0.795 |        0.667 |           0.168 |          0.191 |           0.912 |          2.177 |                0.899 |               2.159 |                     0.721 |                    1.61  |                1 |
| C      | resnet1d | accuracy-only                  |     0.7 | ws_fp32     |  0.091 |      0.634 |        0.564 |           0.159 |          0.393 |           0.358 |          0.661 |                0.472 |               1.252 |                     0.691 |                    1.549 |              nan |
| C      | resnet1d | trust-aware (src CP)           |     0   | ws_fp32     |  1     |      0.642 |        0.576 |           0.289 |          0.347 |           0.321 |          0.719 |                0.439 |               1.128 |                     0.821 |                    1.993 |                1 |
| C      | resnet1d | trust-aware (mondrian CP)      |     0   | ws_fp32     |  1     |      0.642 |        0.576 |           0.289 |          0.347 |           0.321 |          0.719 |                0.439 |               1.128 |                     0.821 |                    1.993 |                1 |
| C      | resnet1d | trust-aware (lobo_mondrian CP) |     0   | ws_fp32     |  1     |      0.642 |        0.576 |           0.289 |          0.347 |           0.321 |          0.719 |                0.439 |               1.128 |                     0.821 |                    1.993 |                1 |
| C      | wdcnn    | accuracy-only                  |     0   | ws_fp32     |  0.023 |      0.639 |        0.513 |           0.221 |          0.432 |           0.109 |          0.75  |                0.671 |               1.425 |                     0.794 |                    1.938 |              nan |
| C      | wdcnn    | trust-aware (src CP)           |     0   | ws_fp32     |  0.023 |      0.639 |        0.513 |           0.221 |          0.432 |           0.109 |          0.75  |                0.671 |               1.425 |                     0.794 |                    1.938 |                1 |
| C      | wdcnn    | trust-aware (mondrian CP)      |     0   | ws_fp32     |  0.023 |      0.639 |        0.513 |           0.221 |          0.432 |           0.109 |          0.75  |                0.671 |               1.425 |                     0.794 |                    1.938 |                1 |
| C      | wdcnn    | trust-aware (lobo_mondrian CP) |     0   | ws_fp32     |  0.023 |      0.639 |        0.513 |           0.221 |          0.432 |           0.109 |          0.75  |                0.671 |               1.425 |                     0.794 |                    1.938 |                1 |
