# SOLQ: Segmenting Objects by Learning Queries

</div>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/solq-segmenting-objects-by-learning-queries/instance-segmentation-on-coco-minival)](https://paperswithcode.com/sota/instance-segmentation-on-coco-minival?p=solq-segmenting-objects-by-learning-queries)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/solq-segmenting-objects-by-learning-queries/instance-segmentation-on-coco)](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=solq-segmenting-objects-by-learning-queries)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/solq-segmenting-objects-by-learning-queries/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=solq-segmenting-objects-by-learning-queries)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/solq-segmenting-objects-by-learning-queries/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=solq-segmenting-objects-by-learning-queries)

</div>

This repository is an official implementation of the NeurIPS 2021 paper [SOLQ: Segmenting Objects by Learning Queries](https://arxiv.org/pdf/2106.02351.pdf).

## Introduction

**TL; DR.** SOLQ is an end-to-end instance segmentation framework with Transformer. It directly outputs the instance masks without any box dependency.

<div style="align: center">
<img src=./figs/solq.png/>
</div>

**Abstract.** In this paper, we propose an end-to-end framework for instance segmentation. Based on the recently introduced DETR, our method, termed SOLQ, segments objects by learning unified queries. In SOLQ, each query represents one object and has multiple representations: class, location and mask. The object queries learned perform classification, box regression and mask encoding simultaneously in an unified vector form. During training phase, the mask vectors encoded are supervised by the compression coding of raw spatial masks. In inference time, mask vectors produced can be directly transformed to spatial masks by the inverse process of compression coding. Experimental results show that SOLQ can achieve state-of-the-art performance, surpassing most of existing approaches. Moreover, the joint learning of unified query representation can greatly improve the detection performance of original DETR. We hope our SOLQ can serve as a strong baseline for the Transformer-based instance segmentation.


## Updates
- (12/10/2021) Release D-DETR+SQR log.txt in [SQR](https://github.com/megvii-research/SOLQ/blob/main/docs/sqr_baseline_r50_log.txt).
- (29/09/2021) Our SOLQ has been accepted by NeurIPS 2021.
- (14/07/2021) Higher performance (Box AP=56.5, Mask AP=46.7) is reported by training with long side 1536 on Swin-L backbone, instead of long side 1333. 

## Main Results

|  **Method**  | **Backbone** | **Dataset**  |  **Box AP**  |  **Mask AP**  |  **Model**  |
|:------:|:------:|:------:|:------:|:------:| :------:| 
| SOLQ | R50 | test-dev | 47.8 | 39.7 | [google](https://drive.google.com/file/d/1D43QroYz2CH3rHDVE54tlByq6dSbmXJK/view?usp=sharing) |
| SOLQ | R101 | test-dev | 48.7 | 40.9 | [google](https://drive.google.com/file/d/1hdHnNDeLP932ZueKEvm5o8T1MwwhP_wm/view?usp=sharing) |
| SOLQ | Swin-L | test-dev | 55.4 | 45.9 | [google](https://drive.google.com/file/d/13Tjf2a81rPTRdtGIQr6y4t-HocoU_bM1/view?usp=sharing) |
| SOLQ | Swin-L & 1536 | test-dev | 56.5 | 46.7 | [google](https://drive.google.com/file/d/17g8NzFUwdbic9e24tS6I1bb4klas_g8c/view?usp=sharing) |


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

#### Visualization on COCO 2017 val dataset

You can visualize on image as follows:

```bash
EXP_DIR=/path/to/checkpoint
python visual.py \
       --meta_arch solq \
       --backbone resnet50 \
       --with_vector \
       --with_box_refine \
       --masks \
       --batch_size 2 \
       --vector_hidden_dim 1024 \
       --vector_loss_coef 3 \
       --output_dir ${EXP_DIR} \
       --resume ${EXP_DIR}/solq_r50_final.pth \
       --eval    
```

## Citing SOLQ
If you find SOLQ useful in your research, please consider citing:
```bibtex
@article{dong2021solq,
  title={SOLQ: Segmenting Objects by Learning Queries},
  author={Dong, Bin and Zeng, Fangao and Wang, Tiancai and Zhang, Xiangyu and Wei, Yichen},
  journal={NeurIPS},
  year={2021}
}
```
