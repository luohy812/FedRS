# FedRS: Federated Learning Under Reliable Supervision for Multi-Organ Segmentation With Inconsistent Labels

This is the implementation of FedRS (Federated Learning Under Reliable Supervision for Multi-Organ Segmentation With Inconsistent Labels). <mark>(This repository is building....)</mark>

<img src="/images/method.jpg">
Our code framework is designed following the structure of [FedMENU (TMI-2023)](https://github.com/DIAL-RPI/Fed-MENU).

## Requirements

1. PyTorch-2.0.1

## Datasets
Eight datasets are used for comparison, including LITS,KITS,Pancreas,AMOS,BTCV.
<img src="/images/dataset.jpeg">
The setting of these datasets is referred to the [FedMENU (TMI-2023)](https://ieeexplore.ieee.org/document/10107904).


## Run 

For EMNIST-Letters dataset:

```bash
FL_BLS_CIL_Our(999, 'emnist-letters', 1, 1500, 120, 'sig', 80, 40, 10, 0.01, 0.001, 10, 0.05, true, 1);
```

## Experimental resuts
### 1.Three-client setting result：
<img src="/images/result1.png">

### 2.Five-client setting result
<img src="/images/result2.png">

### 3.Ablation study result
<img src="/images/ablation_study.jpg">
