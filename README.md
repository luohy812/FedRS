# FedRS: Federated Learning Under Reliable Supervision for Multi-Organ Segmentation With Inconsistent Labels

This is the implementation of FedRS (Federated Learning Under Reliable Supervision for Multi-Organ Segmentation With Inconsistent Labels). <mark>(This repository is building....)</mark>

<img src="/images/method.jpg">
Our code framework is designed following the structure of FedMENU (TMI-2023)(https://github.com/DIAL-RPI/Fed-MENU).

## Requirements

1. PyTorch-2.0.1

## Datasets
Five datasets are used for comparison, including LITS, KITS, Pancreas, AMOS, BTCV.
The setting of these datasets is referred to the Fed-MENU(TMI-2023)(https://ieeexplore.ieee.org/document/10107904)
<img src="/images/dataset.jpeg">
LITS：https://competitions.codalab.org/competitions/17094  

KITS：https://kits19.grand-challenge.org/  

Pancreas：http://medicaldecathlon.com/  

AMOS：https://amos22.grand-challenge.org/  

BTCV：https://www.synapse.org/Synapse:syn3193805/wiki/89480  
## Run 

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=25000 main.py

## Experimental resuts
Experimental Setting: our experiments includes two configurations: a three-client setting and a five-client setting. The three-client setting employs the LITS, KITS, and Pancreas datasets. The five-client setting incorporates the Spleen and Gallbladder datasets from the AMOS dataset, along with the datasets used in the three-client setting. 
We perform both in-federation and out-of-federation evaluations. Specifically, we use LITS, KITS, PANCREAS, and AMOS datasets as in-federation datasets. We randomly divide them into a training set, validation set, and test set in a ratio of 6:1:3. For in-federation evaluation, the test sets from all clients are used for evaluation. In our experiments, the dataset BTCV serves as the separate test set for out-of-federation evaluation.

### 1.Three-client setting result：
<img src="/images/result1.png">

### 2.Five-client setting result
<img src="/images/result2.png">

### 3.Ablation study result
<img src="/images/ablation_study.jpg">
