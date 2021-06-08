# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DCT-Mask (https://github.com/aliyun/DCT-Mask)
# Copyright (c) 2021 Alibaba Cloud. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch_dct
import functools
print = functools.partial(print, flush=True)

class ProcessorDCT(object):
    """
    Apply DCT to encode the binary mask, and use the encoded vector as mask representation in instance segmentation.
    """
    def __init__(self, n_keep, gt_mask_len=128):
        """
        vec_dim: the dimension of the encoded vector, int
        mask_size: the resolution of the initial binary mask representaiton.
        """
        self.vec_dim = n_keep
        self.mask_size = gt_mask_len
        self.n_keep = n_keep
        self.gt_mask_len = gt_mask_len
        assert n_keep <= gt_mask_len*gt_mask_len
        self.dct_vector_coords = self.get_dct_vector_coords(r=gt_mask_len)

    def encode(self, masks, dim=None):
        """
        Encode the mask to vector of vec_dim or specific dimention.
        """
        if dim is None:
            dct_vector_coords = self.dct_vector_coords[:self.vec_dim]
        else:
            dct_vector_coords = self.dct_vector_coords[:dim]
        masks = masks.view([-1, self.mask_size, self.mask_size]).to(dtype=float)  # [N, H, W]
        dct_all = torch_dct.dct_2d(masks, norm='ortho')
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_vectors = dct_all[:, xs, ys]  # reshape as vector
        return dct_vectors  # [N, D]

    def decode(self, dct_vectors, dim=None):
        """
        intput: dct_vector numpy [N,dct_dim]
        output: mask_rc mask reconstructed [N, mask_size, mask_size]
        """
        device = dct_vectors.device
        if dim is None:
            dct_vector_coords = self.dct_vector_coords[:self.vec_dim]
        else:
            dct_vector_coords = self.dct_vector_coords[:dim]
            dct_vectors = dct_vectors[:, :dim]

        N = dct_vectors.shape[0]
        dct_trans = torch.zeros([N, self.mask_size, self.mask_size], dtype=dct_vectors.dtype).to(device)
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_trans[:, xs, ys] = dct_vectors
        mask_rc = torch_dct.idct_2d(dct_trans, norm='ortho')  # [N, mask_size, mask_size]
        return mask_rc

    def get_dct_vector_coords(self, r=128):
        """
        Get the coordinates with zigzag order.
        """
        dct_index = []
        for i in range(r):
            if i % 2 == 0:  # start with even number
                index = [(i-j, j) for j in range(i+1)]
                dct_index.extend(index)
            else:
                index = [(j, i-j) for j in range(i+1)]
                dct_index.extend(index)
        for i in range(r, 2*r-1):
            if i % 2 == 0:
                index = [(i-j, j) for j in range(i-r+1,r)]
                dct_index.extend(index)
            else:
                index = [(j, i-j) for j in range(i-r+1,r)]
                dct_index.extend(index)
        dct_idxs = np.asarray(dct_index)
        return dct_idxs
