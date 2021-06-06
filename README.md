# SOLQ: Segmenting Objects by Learning Queries

This repository is an official implementation of the paper [SOLQ: Segmenting Objects by Learning Queries](https://arxiv.org/pdf/2105.03247.pdf).

## Introduction

**TL; DR.** SOLQ is an end-to-end instance segmentation framework with Transformer. It directly outputs the instance masks without any box dependency.

<div style="align: center">
<img src=./figs/solq.png/>
</div>

**Abstract.** In this paper, we propose an end-to-end framework for instance segmentation. Based on the recently introduced DETR \cite{carion2020detr}, our method, termed SOLQ, segments objects by learning unified queries. In SOLQ, each query represents one object and has multiple representations: class, location and mask. The object queries learned perform classification, box regression and mask encoding simultaneously in an unified vector form. During training phase, the mask vectors encoded are supervised by the compression coding of raw spatial masks. In inference time, mask vectors produced can be directly transformed to spatial masks by the inverse process of compression coding. Experimental results show that SOLQ can achieve state-of-the-art performance, surpassing most of existing approaches. Moreover, the joint learning of unified query representation can greatly improve the detection performance of original DETR. We hope our SOLQ can serve as a strong baseline for the Transformer-based instance segmentation.


## Main Results

|  **Method**  | **Backbone** | **Dataset**  |  **Box AP**  |  **Mask AP**  |  
|:------:|:------:|:------:|:------:|:------:| 
| SOLQ | R50 | test-dev | 47.8 | 39.7 |
| SOLQ | R101 | test-dev | 48.7 | 40.9 |
| SOLQ | Swin-L | test-dev | 55.4 | 45.9 |


## Installation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

## Usage

### Dataset preparation

Please download [COCO](https://cocodataset.org/) and organize them as following:

```
mkdir data && cd data
ln -s /path/to/coco coco
```

### Training and Evaluation

#### Training on single node

Training SOLQ on 8 GPUs as following:

```bash 
sh configs/r50_solq_train.sh

```

#### Evaluation

You can download the pretrained model of SOLQ (the link is in "Main Results" session), then run following command to evaluate it on COCO 2017 val dataset:

```bash 
sh configs/r50_solq_eval.sh

```

#### Evaluation on COCO 2017 test-dev dataset

You can download the pretrained model of SOLQ (the link is in "Main Results" session), then run following command to evaluate it on COCO 2017 test-dev dataset (submit to server):

```bash
sh configs/r50_solq_submit.sh

```

## Citing SOLQ
If you find SOLQ useful in your research, please consider citing:
```bibtex
@article{dong2021solq,
  title={SOLQ: Segmenting Objects by Learning Queries},
  author={Bin Dong, Fangao Zeng, Tiancai Wang, Xiangyu Zhang, Yichen Wei},
  journal={arXiv preprint arXiv:2105.03247},
  year={2021}
}
```