# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import os
import numpy as np
import random
import argparse
import torch
import torchvision.transforms.functional as F
import cv2
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
from models import build_model
from util.tool import load_model
from util.plot_utils import COCO_CATEGORIES
from main import get_args_parser
from torch.nn.functional import interpolate
from typing import List
import motmetrics as mm
import shutil
import json
import pycocotools.mask as mask_util
from detectron2.structures import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.layers import paste_masks_in_image
from detectron2.utils.memory import retry_if_cuda_oom


np.random.seed(2020)
COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]


def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None, mask=None):
    tl = 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        if score is not None:
            cv2.putText(img, score, (c1[0], c1[1] + 30), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    # print("c1c2 = {} {}".format(c1, c2))
    if mask is not None:
        v = Visualizer(img, scale=1)
        vis_mask = v.draw_binary_mask(mask[0].cpu().numpy(), color=None, edge_color=None, text=None)
        img = vis_mask.get_image()
    return img


def draw_bboxes(ori_img, bbox, mask=None, offset=(0, 0), cvt_color=False):
    img = ori_img
    for i, box in enumerate(bbox):
        if mask is not None and mask.shape[0] > 0:
            m = mask[i]
        else:
            m = None
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
            label = int(box[5])
        else:
            score = None
            label = None
        # box text and bar
        color = COCO_CATEGORIES[label-1]['color']
        class_name = COCO_CATEGORIES[label-1]['name']
        label_str = '{}@{}'.format(class_name, score)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        # img = plot_one_box([x1, y1, x2, y2], img, color, label, score=score, mask=m)
        # img = plot_one_box([x1, y1, x2, y2], img, color, label_str, score=None, mask=m)
        img = plot_one_box([x1, y1, x2, y2], img, color, None, score=None, mask=m)
    return img


def draw_points(img: np.ndarray, points: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    assert len(points.shape) == 2 and points.shape[1] == 2, 'invalid points shape: {}'.format(points.shape)
    for i, (x, y) in enumerate(points):
        if i >= 300:
            color = (0, 255, 0)
        cv2.circle(img, (int(x), int(y)), 2, color=color, thickness=2)
    return img


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class Detector(object):
    def __init__(self, args, model=None, postprocessors=None, seq_num=2, img_dir=None):

        self.args = args
        self.detr = model
        self.postprocessors = postprocessors
        self.img_dir = img_dir

        self.file_name = seq_num['file_name']
        self.id = seq_num['id']
        self.img_height = 800
        self.img_width = 1333
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.save_path = os.path.join(self.args.output_dir, 'results')
        os.makedirs(self.save_path, exist_ok=True)

    def init_img(self, img):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        return dt_instances[keep]

    @staticmethod
    def visualize_img_with_bbox(img_path, img, dt_instances: Instances, ref_pts=None, gt_boxes=None):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if dt_instances.has('scores'):
            if dt_instances.has('masks'):
                img_show = draw_bboxes(img, np.concatenate(
                    [dt_instances.boxes, dt_instances.scores.reshape(-1, 1), dt_instances.labels.reshape(-1, 1)],
                    axis=-1), dt_instances.masks)
            else:
                img_show = draw_bboxes(img, np.concatenate(
                    [dt_instances.boxes, dt_instances.scores.reshape(-1, 1), dt_instances.labels.reshape(-1, 1)],
                    axis=-1))
        else:
            img_show = draw_bboxes(img, dt_instances.boxes)
        if ref_pts is not None:
            img_show = draw_points(img_show, ref_pts)
        cv2.imwrite(img_path, img_show)

    def detect(self, prob_threshold=0.4, vis=True):

        img = cv2.imread(os.path.join(self.img_dir, self.file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cur_img, ori_img = self.init_img(img)
        outputs = self.detr(cur_img.cuda().float())
        orig_target_sizes = torch.stack([torch.tensor([self.seq_h, self.seq_w]).cuda()])
        results = self.postprocessors['bbox'](outputs, target_sizes=orig_target_sizes)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([torch.tensor([self.img_height, self.img_width]).cuda()])
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        dt_instances = Instances((1, 1))
        dt_instances.boxes = results[0]['boxes']
        dt_instances.scores = results[0]['scores']
        dt_instances.labels = results[0]['labels']
        dt_instances.masks = results[0]['masks']
        dt_instances = dt_instances.to(torch.device('cpu'))
        dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)

        if vis:
            cur_vis_img_path = os.path.join(self.save_path, self.file_name)
            self.visualize_img_with_bbox(cur_vis_img_path, ori_img, dt_instances, ref_pts=None)
        return dt_instances

if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and weights
    detr, _, postprocessors = build_model(args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    detr = load_model(detr, args.resume)
    detr.eval()
    detr = detr.cuda()

    ann_path = './data/coco/annotations/instances_val2017.json'
    img_dir = './data/coco/val2017'

    annos = json.load(open(ann_path, 'r'))
    images = annos['images'][1000:2000]

    for seq_num in tqdm(images):
        det = Detector(args, model=detr, postprocessors=postprocessors, seq_num=seq_num, img_dir=img_dir)
        det.detect(vis=True)

