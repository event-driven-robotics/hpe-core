import json
import cv2
from pathlib import Path
import numpy as np
from os.path import join, isfile
from tqdm import tqdm

data_path = '/home/ggoyal/data/mpii/poses.json'
image_dir = '/home/ggoyal/data/mpii/tos_synthetic_export'
output_path = '/home/ggoyal/data/mpii/poses_norm.json'

with open(data_path, 'r') as f:
    train_label_list = json.loads(f.readlines()[0])

for j in tqdm(range(len(train_label_list))):

    keypoints = train_label_list[j]["keypoints"]
    other_keypoints = train_label_list[j]["other_keypoints"]
    center = train_label_list[j]['center']
    other_centers = train_label_list[j]['other_centers']

    assert len(other_centers) == len(other_keypoints)

    image_path = join(image_dir, train_label_list[j]["img_name"])

    if isfile(image_path):
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        for i in range(len(keypoints)):
            keypoints[i][0] = keypoints[i][0] / w
            keypoints[i][1] = keypoints[i][1] / h
        for a in range(len(other_keypoints)):
            # print(other_keypoints[a])
            other_centers[a] = [other_centers[a][0] / w, other_centers[a][1] / h]
            for b in range(len(other_keypoints[a])):
                other_keypoints[a][b][0] = other_keypoints[a][b][0] / w
                other_keypoints[a][b][1] = other_keypoints[a][b][1] / h
                other_keypoints[a][b].pop()
            other_keypoints_reshaped = []
            for i in range(16):
                other_keypoints_reshaped.append([])
            other_keypoints_reshaped.append(sum(other_keypoints, []))

        center = [center[0] / w, center[1] / h]

    train_label_list[j]["keypoints"] = sum(keypoints, [])
    train_label_list[j]["center"] = center
    train_label_list[j]["other_keypoints"] = other_keypoints_reshaped
    train_label_list[j]["other_center"] = other_centers

with open(Path(output_path).resolve(), 'w') as f:
    json.dump(train_label_list, f, ensure_ascii=False)
