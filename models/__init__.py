# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .deformable_detr import build
from .solq import build as build_solq
from .fast_solq import build as build_fast_solq
from .solq_pca import build as build_solq_pca

def build_model(args):
    if args.meta_arch == 'solq':
        return build_solq(args)
    if args.meta_arch == 'fast_solq':
        return build_fast_solq(args)
    if args.meta_arch == 'solq_pca':
        return build_solq_pca(args)
    return build(args)

