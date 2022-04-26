"""
@Fire
https://github.com/fire717
"""
import gc
import os
import torch
import numpy as np
import cv2
import json
import time

# import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from lib.task.task_tools import getSchedu, getOptimizer, movenetDecode, clipGradient, restore_sizes, image_show
from lib.loss.movenet_loss import MovenetLoss
from lib.utils.utils import printDash, ensure_loc
# from lib.visualization.visualization import superimpose_pose
from lib.utils.metrics import myAcc, pck


class Task():
    def __init__(self, cfg, model):

        self.cfg = cfg
        self.init_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # edit for Franklin
        self.model = model.to(self.device)

        ############################################################
        # loss
        self.loss_func = MovenetLoss(self.cfg)

        # optimizer
        self.optimizer = getOptimizer(self.cfg['optimizer'],
                                      self.model,
                                      self.cfg['learning_rate'],
                                      self.cfg['weight_decay'])

        self.val_losses = np.zeros([20])
        self.early_stop = 0
        self.val_loss_best = np.Inf

        # scheduler
        self.scheduler = getSchedu(self.cfg['scheduler'], self.optimizer)

        # ensure_loc(os.path.join(self.cfg['save_dir'], self.cfg['label']))

    def predict(self, data_loader, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.model.eval()
        correct = 0
        size = self.cfg["img_size"]
        with torch.no_grad():

            for (img, img_name) in data_loader:

                print("Shape after loading:", img.shape)
                img = img.to(self.device)

                output = self.model(img)

                pre = movenetDecode(output, None, mode='output', num_joints=self.cfg["num_classes"])

                basename = os.path.basename(img_name[0])
                img = np.transpose(img[0].cpu().numpy(), axes=[1, 2, 0])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                h, w = img.shape[:2]
                print("Shape for viewing:", img.shape)

                # cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_img.jpg"), img)

                for i in range(len(pre[0]) // 2):
                    x = int(pre[0][i * 2] * w)
                    y = int(pre[0][i * 2 + 1] * h)
                    cv2.circle(img, (x, y), 3, (255, 0, 0), 2)

                image_show(img)
                continue
                cv2.imwrite(os.path.join(save_dir, basename), img)

                # debug
                heatmaps = output[0].cpu().numpy()[0]
                centers = output[1].cpu().numpy()[0]
                regs = output[2].cpu().numpy()[0]
                offsets = output[3].cpu().numpy()[0]

                hm = cv2.resize(np.sum(heatmaps, axis=0), (size, size)) * 255
                cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_heatmaps.jpg"), hm)
                cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_center.jpg"),
                            cv2.resize(centers[0] * 255, (size, size)))
                cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_regs0.jpg"),
                            cv2.resize(regs[0] * 255, (size, size)))

    def predict_online(self, img_in):

        self.model.eval()
        correct = 0
        size = self.cfg["img_size"]
        with torch.no_grad():

            img = torch.from_numpy(img_in)
            img = img.to(self.device)

            img_size_original = img.shape
            img = img
            img = img.to(self.device)
            output = self.model(img)
            instant = {}

            if self.cfg['show_center']:
                centers = output[1].cpu().numpy()[0]
                from lib.utils.utils import maxPoint
                cx, cy = maxPoint(centers)
                instant['center'] = np.array([cx[0][0],cy[0][0]])/centers.shape[1]

            pre = movenetDecode(output, None, mode='output', num_joints=self.cfg["num_classes"])

            img = np.transpose(img[0].cpu().numpy(), axes=[1, 2, 0])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, instant['joints'] = restore_sizes(img, pre, (int(img_size_original[2]), int(img_size_original[3])))
            instant['center_heatmap'] = centers[0]

            return instant


    def evaluate(self, data_loader):
        self.model.eval()

        correct_kps = 0.0
        total_kps = 0.0
        joint_correct = np.zeros([self.cfg["num_classes"]])
        joint_total = np.zeros([self.cfg["num_classes"]])
        size = self.cfg["img_size"]
        text_location = (10, size * 2 - 10)  # bottom left corner of the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        fontColor = (0, 0, 255)
        thickness = 1
        lineType = 2

        with torch.no_grad():
            start = time.time()
            for batch_idx, (imgs, labels, kps_mask, img_names, torso_diameter, head_size_norm) in enumerate(
                    data_loader):

                if batch_idx % 100 == 0 and batch_idx > 10:
                    print('Finished samples: ', batch_idx)
                    acc_intermediate = correct_kps / total_kps
                    acc_joint_mean_intermediate = np.mean(joint_correct / joint_total)
                    print('[Info] Mean Keypoint Acc: {:.3f}%'.format(100. * acc_intermediate))
                    print('[Info] Mean Joint Acc: {:.3f}%'.format(100. * acc_joint_mean_intermediate))
                    print('[Info] Average Freq:', (batch_idx / (time.time() - start)), '\n')

                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs)

                pre = movenetDecode(output, kps_mask, mode='output', num_joints=self.cfg["num_classes"])
                gt = movenetDecode(labels, kps_mask, mode='label', num_joints=self.cfg["num_classes"])

                if self.cfg['dataset'] in ['coco', 'mpii']:
                    pck_acc = pck(pre, gt, head_size_norm, num_classes=self.cfg["num_classes"], mode='head')
                    th_val = head_size_norm
                else:
                    pck_acc = pck(pre, gt, torso_diameter, num_classes=self.cfg["num_classes"], mode='torso')
                    th_val = torso_diameter

                correct_kps += pck_acc["total_correct"]
                total_kps += pck_acc["total_keypoints"]
                joint_correct += pck_acc["correct_per_joint"]
                joint_total += pck_acc["anno_keypoints_per_joint"]

                img = np.transpose(imgs[0].cpu().numpy(), axes=[1, 2, 0])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                h, w = img.shape[:2]

                for i in range(len(gt[0]) // 2):
                    x = int(gt[0][i * 2] * w)
                    y = int(gt[0][i * 2 + 1] * h)
                    cv2.circle(img, (x, y), 2, (0, 255, 0), 1)  # gt keypoints in green

                    x = int(pre[0][i * 2] * w)
                    y = int(pre[0][i * 2 + 1] * h)
                    cv2.circle(img, (x, y), 2, (0, 0, 255), 1)  # predicted keypoints in red

                img2 = cv2.resize(img, (size * 2, size * 2), interpolation=cv2.INTER_LINEAR)
                str = "acc: %.2f, th: %.2f " % (pck_acc["total_correct"] / pck_acc["total_keypoints"], th_val)
                cv2.putText(img2, str,
                            text_location,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                # cv2.line(img2, [10, 10], [10 + int(head_size_norm * 2), 10], [0, 0, 255], 3)
                basename = os.path.basename(img_names[0])
                ensure_loc(self.cfg['eval_outputs'])
                cv2.imwrite(os.path.join(self.cfg['eval_outputs'], basename), img)

        acc = correct_kps / total_kps
        acc_joint_mean = np.mean(joint_correct / joint_total)
        print('[Info] Mean Keypoint Acc: {:.3f}%'.format(100. * acc))
        print('[Info] Mean Joint Acc: {:.3f}% \n'.format(100. * acc_joint_mean))

    def infer_video(self, data_loader, video_path):
        self.model.eval()

        correct_kps = 0.0
        total_kps = 0.0
        joint_correct = np.zeros([self.cfg["num_classes"]])
        joint_total = np.zeros([self.cfg["num_classes"]])
        size = self.cfg["img_size"]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (size * 2, size * 2))

        text_location = (10, size * 2 - 10)  # bottom left corner of the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        fontColor = (0, 0, 255)
        thickness = 1
        lineType = 2

        with torch.no_grad():
            start = time.time()
            for batch_idx, (imgs, labels, kps_mask, img_names, torso_diameter, head_size_norm) in enumerate(
                    data_loader):

                if batch_idx % 100 == 0 and batch_idx > 10:
                    print('Finished samples: ', batch_idx)
                    acc_intermediate = correct_kps / total_kps
                    acc_joint_mean_intermediate = np.mean(joint_correct / joint_total)
                    # print('[Info] Mean Keypoint Acc: {:.3f}%'.format(100. * acc_intermediate))
                    print('[Info] Mean Joint Acc: {:.3f}%'.format(100. * acc_joint_mean_intermediate))
                    print('[Info] Average Freq:', (batch_idx / (time.time() - start)), '\n')

                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs)

                pre = movenetDecode(output, kps_mask, mode='output', num_joints=self.cfg["num_classes"])
                gt = movenetDecode(labels, kps_mask, mode='label', num_joints=self.cfg["num_classes"])

                if self.cfg['dataset'] in ['coco', 'mpii']:
                    pck_acc = pck(pre, gt, head_size_norm, num_classes=self.cfg["num_classes"], mode='head')
                    th_val = head_size_norm
                else:
                    pck_acc = pck(pre, gt, torso_diameter, num_classes=self.cfg["num_classes"], mode='torso')
                    th_val = torso_diameter

                correct_kps += pck_acc["total_correct"]
                total_kps += pck_acc["total_keypoints"]
                joint_correct += pck_acc["correct_per_joint"]
                joint_total += pck_acc["anno_keypoints_per_joint"]

                img = np.transpose(imgs[0].cpu().numpy(), axes=[1, 2, 0])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                h, w = img.shape[:2]

                for i in range(len(gt[0]) // 2):
                    # x = int(gt[0][i * 2] * w)
                    # y = int(gt[0][i * 2 + 1] * h)
                    # cv2.circle(img, (x, y), 2, (0, 255, 0), 1)  # gt keypoints in green

                    x = int(pre[0][i * 2] * w)
                    y = int(pre[0][i * 2 + 1] * h)
                    cv2.circle(img, (x, y), 2, (0, 0, 255), 1)  # predicted keypoints in red

                img2 = cv2.resize(img, (size * 2, size * 2), interpolation=cv2.INTER_LINEAR)
                # str = "acc: %.2f, th: %.2f " % (pck_acc["total_correct"] / pck_acc["total_keypoints"], th_val)
                # cv2.putText(img2, str,
                #             text_location,
                #             font,
                #             fontScale,
                #             fontColor,
                #             thickness,
                #             lineType)
                # cv2.line(img2, [10, 10], [10 + int(head_size_norm * 2), 10], [0, 0, 255], 3)
                # cv2.imshow("prediction", img)
                # cv2.waitKey(10)
                # basename = os.path.basename(img_names[0])
                # ensure_loc('eval_result')
                # cv2.imwrite(os.path.join('eval_result', basename), img)
                img2 = np.uint8(img2)
                out.write(img2)

        acc = correct_kps / total_kps
        acc_joint_mean = np.mean(joint_correct / joint_total)
        print('[Info] Mean Keypoint Acc: {:.3f}%'.format(100. * acc))
        print('[Info] Mean Joint Acc: {:.3f}% \n'.format(100. * acc_joint_mean))
        out.release()

    def modelLoad(self, model_path, data_parallel=False):

        if os.path.splitext(model_path)[-1] == '.json':
            with open(model_path, 'r') as f:
                model_path = json.loads(f.readlines()[0])
                str1 = ''
            init_epoch = int(str1.join(os.path.basename(model_path).split('_')[0][1:]))
            self.init_epoch = init_epoch
        print(model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        if data_parallel:
            self.model = torch.nn.DataParallel(self.model)
