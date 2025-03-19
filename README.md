# FedCBLS: Federated Class Incremental Learning Method with High Accuracy and Extremely Low Communication Cost Based on Broad Learning System

This is the implementation of FedCBLS (Federated Class Incremental Learning Method with High Accuracy and Extremely Low Communication Cost Based on Broad Learning System). <mark>(This repository is building....)</mark>


< img src="fedcbls.png" height="600" width="900" >

## Requirements

1. MatLab 2021a
2. PyTorch-1.8.0

## Datasets
Eight datasets are used for comparison, including MNIST, EMNIST-Letters, EMNIST-Balanced, CIFAR-10, EMNIST-LTP, EMNIST-shuffle, CIFAR-100, and MNIST-SVHN-F.

The setting of these datasets is referred to the [FedCIL (ICLR-2023)](https://iclr.cc/virtual/2023/poster/11660) and [AF-FCL (ICLR-2023)](https://iclr.cc/virtual/2024/poster/18593).


## Run 

For EMNIST-Letters dataset:

```bash
FL_BLS_CIL_Our(999, 'emnist-letters', 1, 1500, 120, 'sig', 80, 40, 10, 0.01, 0.001, 10, 0.05, true, 1);
```

## Experimental resuts

< img src="result.png" height="807" width="900" >
