# FedRS: Federated Learning Under Reliable Supervision for Multi-Organ Segmentation With Inconsistent Labels

This is the implementation of FedRS (Federated Learning Under Reliable Supervision for Multi-Organ Segmentation With Inconsistent Labels). <mark>(This repository is building....)</mark>


< img src="fedcbls.png" height="600" width="900" >

## Requirements

1. PyTorch-2.0.1

## Datasets
Eight datasets are used for comparison, including LITS,KITS,Pancreas,AMOS,BTCV.

The setting of these datasets is referred to the [FedMENU (TMI-2023)](https://ieeexplore.ieee.org/document/10107904).


## Run 

For EMNIST-Letters dataset:

```bash
FL_BLS_CIL_Our(999, 'emnist-letters', 1, 1500, 120, 'sig', 80, 40, 10, 0.01, 0.001, 10, 0.05, true, 1);
```

## Experimental resuts
<img src="result1.png" height="2000" width="2299">

