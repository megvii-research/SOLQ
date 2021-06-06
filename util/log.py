# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os

from loguru import logger
from tensorboardX import SummaryWriter


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        mode(str): log file write mode, `append` or `override`. default is `a`.
    Return:
        logger instance.
    """
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    if distributed_rank > 0:
        logger.remove()
    logger.add(
        save_file, format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", filter="", level="INFO", enqueue=True
    )

    return logger


def setup_writer(filename, distributed_rank):
    if distributed_rank > 0:
        writer = None
    else:
        writer = SummaryWriter(filename)
    return writer
