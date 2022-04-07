"""
@Fire
https://github.com/fire717
"""
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data.dataset import Dataset

import random
import cv2
import albumentations as A
import json
import platform
import math

from lib.data.data_augment import DataAug
from lib.utils.utils import maxPoint, extract_keypoints


def getFileNames(file_dir, tail_list=['.png', '.jpg', '.JPG', '.PNG']):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in tail_list:
                L.append(os.path.join(root, file))
    return L


def label2heatmap(keypoints, other_keypoints, img_size):
    # keypoints: target person
    # other_keypoints: other people's keypoints need to be add to the heatmap
    heatmaps = []
    # print(len(keypoints))

    keypoints_range = np.reshape(keypoints, (-1, 3))
    keypoints_range = keypoints_range[keypoints_range[:, 2] > 0]
    # print(keypoints_range)

    min_x = np.min(keypoints_range[:, 0])
    min_y = np.min(keypoints_range[:, 1])
    max_x = np.max(keypoints_range[:, 0])
    max_y = np.max(keypoints_range[:, 1])
    area = (max_y - min_y) * (max_x - min_x)
    sigma = 3
    if area < 0.16:
        sigma = 3
    elif area < 0.3:
        sigma = 5
    else:
        sigma = 7

    for i in range(0, len(keypoints), 3):
        if keypoints[i + 2] == 0:  # If the keypoint is not annotated
            heatmaps.append(np.zeros((img_size // 4, img_size // 4)))
            continue

        x = int(keypoints[i] * img_size // 4)  # 取值应该是0-47
        y = int(keypoints[i + 1] * img_size // 4)
        if x == img_size // 4: x = (img_size // 4 - 1)
        if y == img_size // 4: y = (img_size // 4 - 1)
        if x > img_size // 4 or x < 0: x = -1
        if y > img_size // 4 or y < 0: y = -1
        heatmap = generate_heatmap(x, y, other_keypoints[i // 3], (img_size // 4, img_size // 4), sigma)

        heatmaps.append(heatmap)

    heatmaps = np.array(heatmaps, dtype=np.float32)

    return heatmaps, sigma


def label2center(cx, cy, other_centers, img_size, sigma):
    heatmaps = []

    heatmap = generate_heatmap(cx, cy, other_centers, (img_size // 4, img_size // 4), sigma + 2)
    heatmaps.append(heatmap)

    heatmaps = np.array(heatmaps, dtype=np.float32)
    # print(heatmaps.shape)

    return heatmaps


def label2reg(keypoints, cx, cy, img_size):
    heatmaps = np.zeros((len(keypoints) // 3 * 2, img_size // 4, img_size // 4), dtype=np.float32)
    for i in range(len(keypoints) // 3):
        if keypoints[i * 3 + 2] == 0:
            continue

        x = keypoints[i * 3] * img_size // 4
        y = keypoints[i * 3 + 1] * img_size // 4
        if x == img_size // 4: x = (img_size // 4 - 1)
        if y == img_size // 4: y = (img_size // 4 - 1)
        if x > img_size // 4 or x < 0 or y > img_size // 4 or y < 0:
            continue

        reg_x = x - cx
        reg_y = y - cy

        for j in range(cy - 2, cy + 3):
            if j < 0 or j > img_size // 4 - 1:
                continue
            for k in range(cx - 2, cx + 3):
                if k < 0 or k > img_size // 4 - 1:
                    continue
                if cx < img_size // 4 / 2 - 1:
                    heatmaps[i * 2][j][k] = reg_x - (cx - k)  # /(img_size//4)
                else:
                    heatmaps[i * 2][j][k] = reg_x + (cx - k)  # /(img_size//4)
                if cy < img_size // 4 / 2 - 1:
                    heatmaps[i * 2 + 1][j][k] = reg_y - (cy - j)  # /(img_size//4)
                else:
                    heatmaps[i * 2 + 1][j][k] = reg_y + (cy - j)
    # bb
    return heatmaps


def label2offset(keypoints, cx, cy, regs, img_size):
    heatmaps = np.zeros((len(keypoints) // 3 * 2, img_size // 4, img_size // 4), dtype=np.float32)
    for i in range(len(keypoints) // 3):
        if keypoints[i * 3 + 2] == 0:
            continue

        large_x = int(keypoints[i * 3] * img_size)
        large_y = int(keypoints[i * 3 + 1] * img_size)

        small_x = int(regs[i * 2, cy, cx] + cx)
        small_y = int(regs[i * 2 + 1, cy, cx] + cx)

        offset_x = large_x / 4 - small_x
        offset_y = large_y / 4 - small_y

        if small_x == img_size // 4: small_x = (img_size // 4 - 1)
        if small_y == img_size // 4: small_y = (img_size // 4 - 1)
        if small_x > img_size // 4 or small_x < 0 or small_y > img_size // 4 or small_y < 0:
            continue

        heatmaps[i * 2][small_y][small_x] = offset_x  # /(img_size//4)
        heatmaps[i * 2 + 1][small_y][small_x] = offset_y  # /(img_size//4)

    return heatmaps


def generate_heatmap(x, y, other_keypoints, size, sigma):
    sigma += 6
    heatmap = np.zeros(size)
    if x < 0 or y < 0 or x >= size[0] or y >= size[1]:
        return heatmap

    tops = [[x, y]]
    if len(other_keypoints) > 0:
        # add other people's keypoints
        for i in range(len(other_keypoints)):
            x = int(other_keypoints[i][0] * size[0])
            y = int(other_keypoints[i][1] * size[1])
            if x == size[0]: x = (size[0] - 1)
            if y == size[1]: y = (size[1] - 1)
            if x > size[0] or x < 0 or y > size[1] or y < 0: continue
            tops.append([x, y])

    for top in tops:
        x, y = top
        x0 = max(0, x - sigma // 2)
        x1 = min(size[0], x + sigma // 2)
        y0 = max(0, y - sigma // 2)
        y1 = min(size[1], y + sigma // 2)

        for map_y in range(y0, y1):
            for map_x in range(x0, x1):
                d2 = ((map_x - x) ** 2 + (map_y - y) ** 2) ** 0.5

                if d2 <= sigma // 2:
                    heatmap[map_y, map_x] += math.exp(-d2 / (sigma // 2) * 3)
                if heatmap[map_y, map_x] > 1:
                    heatmap[map_y, map_x] = 1
    return heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def generate_heatmap1(x, y, other_keypoints, size, sigma):
    sigma += 4

    k = 1
    heatmap = np.zeros(size)
    height, width = size
    diameter = sigma  # 2 * radius + 1
    radius = sigma // 2
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    tops = [[x, y]]

    for top in tops:
        x, y = top

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    heatmap[heatmap > 1] = 1
    return heatmap


def generate_heatmap3(x, y, other_keypoints, size, sigma):

    tops = [[x, y]]
    if len(other_keypoints) > 0:
        # add other people's keypoints
        for i in range(len(other_keypoints)):
            x = int(other_keypoints[i][0] * size[0])
            y = int(other_keypoints[i][1] * size[1])
            if x == size[0]: x = (size[0] - 1)
            if y == size[1]: y = (size[1] - 1)
            if x > size[0] or x < 0 or y > size[1] or y < 0: continue
            tops.append([x, y])

    target = np.zeros((size[1], size[0]), dtype=np.float32)

    for top in tops:
        x, y = top

        ul = [int(x - sigma), int(y - sigma)]
        br = [int(x + sigma + 1), int(y + sigma + 1)]

        # # Generate gaussian
        sigma2 = 2 * sigma + 1  # 直径
        x = np.arange(0, sigma2, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = sigma2 // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma / 3) ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], size[0])
        img_y = max(0, ul[1]), min(br[1], size[1])

        target[img_y[0]:img_y[1], img_x[0]:img_x[1]] += g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    target[target > 1] = 1
    return target


def get_headsize(head_size_scaled, img_size):
    head_size = (head_size_scaled * float(img_size // 4))
    return head_size

def get_torso_diameter(keypoints):
    kps = np.reshape(keypoints,[-1,3])[:,:-1]
    if len(keypoints)//3 == 13:
        left_hip = kps[7,:]
        right_hip = kps[8,:]
        left_shoulder = kps[1,:]
        right_shoulder = kps[2,:]
        hip_width = math.dist(left_hip,right_hip)
        shoulder_width = math.dist(left_shoulder,right_shoulder)
        torso_diameter = np.mean([hip_width,shoulder_width])
        return torso_diameter
    else:
        return 0

######## dataloader
class TensorDataset(Dataset):
    def __init__(self, data_labels, img_dir, img_size, data_aug=None, num_classes=13):
        self.data_labels = data_labels
        self.img_dir = img_dir
        self.data_aug = data_aug
        self.img_size = img_size
        self.num_classes = num_classes
        self.interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
                               cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]

    def __getitem__(self, index):
        item = self.data_labels[index]

        if len(item['other_keypoints']) == 0:
            item['other_keypoints'] = [[] for i in range(13)]
        """
        item = {
                     "img_name":save_name,
                     "head_size":head_size,
                     "head_size_scaled":head_size_scaled,
                     "keypoints":save_keypoints,
                     "center":save_center,
                     "other_centers":other_centers,
                     "other_keypoints":other_keypoints,
                    }
        """
        img_path = os.path.join(self.img_dir, item["img_name"])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=random.choice(self.interp_methods))
        #### Data Augmentation
        if self.data_aug is not None:
            img, item = self.data_aug(img, item)
        img = img.astype(np.float32)
        img = np.transpose(img, axes=[2, 0, 1])
        head_size_scaled = item.get("head_size_scaled", None)
        keypoints = item["keypoints"]
        center = item['center']
        other_centers = item["other_centers"]
        other_keypoints = item["other_keypoints"]

        kps_mask = np.ones(len(keypoints) // 3)
        for i in range(len(keypoints) // 3):
            if keypoints[i * 3 + 2] == 0:
                kps_mask[i] = 0
        heatmaps, sigma = label2heatmap(keypoints, other_keypoints, self.img_size)  # (17, 48, 48)
        cx = min(max(0, int(center[0] * self.img_size // 4)), self.img_size // 4 - 1)
        cy = min(max(0, int(center[1] * self.img_size // 4)), self.img_size // 4 - 1)
        centers = label2center(cx, cy, other_centers, self.img_size, sigma)  # (1, 48, 48)
        regs = label2reg(keypoints, cx, cy, self.img_size)  # (14, 48, 48)
        offsets = label2offset(keypoints, cx, cy, regs, self.img_size)  # (14, 48, 48)
        n = self.num_classes
        labels = np.concatenate([heatmaps[:n, :, :], centers, regs[:2 * n, :, :], offsets[:2 * n, :, :]], axis=0)
        torso_diameter = get_torso_diameter(keypoints)

        return img, labels, kps_mask, img_path, torso_diameter, head_size_scaled

    def __len__(self):
        return len(self.data_labels)


class TensorDatasetTest(Dataset):

    def __init__(self, data_labels, img_dir, img_size, data_aug=None):
        self.data_labels = data_labels
        self.img_dir = img_dir
        self.data_aug = data_aug
        self.img_size = img_size

        self.interp_methods = cv2.INTER_LINEAR

    def __getitem__(self, index):
        img_name = self.data_labels[index]

        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=self.interp_methods)

        img = img.astype(np.float32)
        img = np.transpose(img, axes=[2, 0, 1])

        return img, img_name

    def __len__(self):
        return len(self.data_labels)


###### get data loader
def getDataLoader(mode, input_data, cfg):
    if mode == "trainval":
        pass

    elif mode == "val":

        pass

    elif mode == "eval":

        val_loader = torch.utils.data.DataLoader(
            TensorDataset(input_data[0],
                          cfg['eval_img_path'],
                          cfg['img_size'],
                          num_classes=cfg['num_classes']
                          ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False)

        return val_loader
    elif mode == "test":

        data_loader = torch.utils.data.DataLoader(
            TensorDatasetTest(input_data,
                              cfg['test_img_path'],
                              cfg['img_size'],
                              ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False)

        return data_loader
    else:
        raise Exception("Unknown mode.")
