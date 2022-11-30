#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:12:04 2022

convert instance labels to three class labels:
0: background
1: interior
2: boundary
@author: jma
"""

import os

join = os.path.join
import argparse

from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm


def normalize(img_data):
    pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
    for i in range(3):
        img_channel_i = img_data[:, :, i]
        if len(img_channel_i[np.nonzero(img_channel_i)]) > 0:
            pre_img_data[:, :, i] = normalize_channel(img_channel_i, lower=1, upper=99)
    return pre_img_data


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser('Preprocessing for microscopy image segmentation', add_help=False)
    parser.add_argument('-i', '--input_path', default='./data/nips/Train_Labeled', type=str,
                        help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='./data/nips/Train_Pre_3class_slide', type=str,
                        help='preprocessing data path')
    args = parser.parse_args()

    source_path = args.input_path
    target_path = args.output_path

    img_path = join(source_path, 'images')
    gt_path = join(source_path, 'labels')

    img_names = sorted(os.listdir(img_path))
    gt_names = [img_name.split('.')[0] + '.tif' for img_name in img_names]

    pre_img_path = join(target_path, 'images')
    pre_gt_path = join(target_path, 'labels')
    os.makedirs(pre_img_path, exist_ok=True)
    os.makedirs(pre_gt_path, exist_ok=True)

    for img_name, gt_name in zip(tqdm(img_names), gt_names):
        if img_name.endswith('.tif') or img_name.endswith('.tiff'):
            img_data = tif.imread(join(img_path, img_name))
        else:
            img_data = io.imread(join(img_path, img_name))

        # normalize image data
        if len(img_data.shape) == 2:
            img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
        elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
            img_data = img_data[:, :, :3]
        else:
            pass

        win_size = 1536
        stride = 1024
        total_col = max(int((img_data.shape[1] - win_size + stride - 1) / stride) + 1, 1)
        total_row = max(int((img_data.shape[0] - win_size + stride - 1) / stride) + 1, 1)
        gt_data = tif.imread(join(gt_path, gt_name))
        for row in range(total_row):
            for col in range(total_col):
                window = img_data[row*stride:row*stride+win_size, col*stride:col*stride+win_size, :]
                window_gt = gt_data[row*stride:row*stride+win_size, col*stride:col*stride+win_size]
                window = normalize(window)
                # to conduct augmentation, you just need to uncomment commented code to replace the following code
                io.imsave(join(target_path, 'images',
                               img_name.split('.')[0] + '(%d,%d).png' % (row, col)),
                          window.astype(np.uint8))
                tif.imwrite(join(target_path, 'labels',
                                 img_name.split('.')[0] + '(%d,%d).tif' % (row, col)),
                            window_gt)
                # for i in range(4):
                #     rot_window = np.rot90(window, i)
                #     rot_window_gt = np.rot90(window_gt, i)
                #     io.imsave(join(target_path, 'images',
                #                    img_name.split('.')[0] + '(%d,%d)_r%df%d.png' % (row, col, i, 0)),
                #               rot_window.astype(np.uint8))
                #     tif.imwrite(join(target_path, 'labels',
                #                      img_name.split('.')[0] + '(%d,%d)_r%df%d.tif' % (row, col, i, 0)),
                #                 rot_window_gt)
                # window = np.flip(window, 1)
                # window_gt = np.flip(window_gt, 1)
                # for i in range(4):
                #     rot_window = np.rot90(window, i)
                #     rot_window_gt = np.rot90(window_gt, i)
                #     io.imsave(join(target_path, 'images',
                #                    img_name.split('.')[0] + '(%d,%d)_r%df%d.png' % (row, col, i, 1)),
                #               rot_window.astype(np.uint8))
                #     tif.imwrite(join(target_path, 'labels',
                #                      img_name.split('.')[0] + '(%d,%d)_r%df%d.tif' % (row, col, i, 1)),
                #                 rot_window_gt)




if __name__ == "__main__":
    main()























