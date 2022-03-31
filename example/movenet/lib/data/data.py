"""
@Fire
https://github.com/fire717
"""
import os
import random
import numpy as np
import json
import copy

from lib.data.data_tools import getDataLoader, getFileNames
from lib.task.task_tools import movenetDecode


class Data():
    def __init__(self, cfg):

        self.cfg = cfg

    def dataBalance(self, data_list):
        """
        item = {
                     "img_name":save_name,
                     "keypoints":save_keypoints,
                     [左手腕，左手肘，左肩，头，右肩，右手肘，右手腕]
                     "center":save_center,
                     "other_centers":other_centers,
                     "other_keypoints":other_keypoints,
                    }
        """
        new_data_list = copy.deepcopy(data_list)

        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for item in data_list:
            keypoints = item['keypoints']

            kpt_np = np.array(keypoints).reshape((-1, 3))
            kpt_np_valid = kpt_np[kpt_np[:, 2] > 0]

            w = np.max(kpt_np_valid[:, 0]) - np.min(kpt_np_valid[:, 0])
            h = np.max(kpt_np_valid[:, 1]) - np.min(kpt_np_valid[:, 1])

            if (kpt_np[1][1] > kpt_np[0][1] and kpt_np[1][1] > kpt_np[2][1]) or \
                    (kpt_np[5][1] > kpt_np[6][1] and kpt_np[5][1] > kpt_np[4][1]):
                count1 += 1
                print(item)
                for i in range(2):
                    new_data_list.append(item)

            if kpt_np[2][0] - kpt_np[4][0] < \
                    max(kpt_np[1][1] - kpt_np[2][1], kpt_np[5][1] - kpt_np[4][1]):
                count2 += 1

                for i in range(2):
                    new_data_list.append(item)

            if (kpt_np[1][1] < kpt_np[2][1]) or \
                    (kpt_np[5][1] < kpt_np[4][1]):
                count3 += 1

                for i in range(5):
                    new_data_list.append(item)

            if h < w:
                count4 += 1
                for i in range(3):
                    new_data_list.append(item)

        print(count1, count2, count3, count4)

        random.shuffle(new_data_list)
        return new_data_list


    def getEvalDataloader(self):
        with open(self.cfg['eval_label_path'], 'r') as f:
            data_label_list = json.loads(f.readlines()[0])

        print("[INFO] Total images: ", len(data_label_list))

        input_data = [data_label_list]
        data_loader = getDataLoader("eval",
                                    input_data,
                                    self.cfg)
        return data_loader

    def getTestDataloader(self):
        data_names = getFileNames(self.cfg['test_img_path'])
        test_loader = getDataLoader("test",
                                    data_names,
                                    self.cfg)
        return test_loader
