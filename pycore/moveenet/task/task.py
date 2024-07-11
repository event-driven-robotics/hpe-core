
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
import csv

# import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from pycore.moveenet.task.task_tools import getSchedu, getOptimizer, movenetDecode, clipGradient, restore_sizes, image_show
from pycore.moveenet.loss.movenet_loss import MovenetLoss
from pycore.moveenet.utils.utils import printDash, ensure_loc
from pycore.moveenet.visualization.visualization import superimpose_pose, hpecore_kps_labels
from pycore.moveenet.utils.metrics import myAcc, pck
from datasets.h36m.utils.parsing import movenet_to_hpecore


class Task():
    def __init__(self, cfg, model):

        self.cfg = cfg
        self.init_epoch = 0
        if(self.cfg["GPU_ID"]):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print(self.device)
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

                # image_show(img)
                # continue
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

    def predict_online(self, img_in, write_csv=None, ts = None):

        self.model.eval()
        correct = 0
        size = self.cfg["img_size"]
        with torch.no_grad():

            img_size_original = img_in.shape
            input_image_resized = np.zeros([1, 3, size, size])
            # print(input_image_resized.shape)

            input_image = cv2.resize(img_in, (size, size))
            input_image_resized[0, 0, :, :] = input_image[:, :]
            input_image_resized[0, 1, :, :] = input_image[:, :]
            input_image_resized[0, 2, :, :] = input_image[:, :]
            input_image_resized = input_image_resized.astype(np.float32)

            img = torch.from_numpy(input_image_resized)
            img = img.to(self.device)

            # img_size_original = img.shape
            img = img.to(self.device)
            start_sample = time.time()
            output = self.model(img)
            instant = {}

            try:

                if self.cfg['show_center']:
                    centers = output[1].cpu().numpy()[0]
                    from pycore.moveenet.utils.utils import maxPoint
                    cx, cy = maxPoint(centers)
                    instant['center'] = np.array([cx[0][0],cy[0][0]])/centers.shape[1]
            except KeyError:
                pass
            pre = movenetDecode(output, None, mode='occlusion', num_joints=self.cfg["num_classes"])
            if self.cfg['num_classes'] == 7:
                x = np.resize([0],[1,18])
                pre = np.concatenate((pre, x), axis=1)
            # img = np.transpose(img[0].cpu().numpy(), axes=[1, 2, 0])
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, pose_pre = restore_sizes(img[0], pre, (int(img_size_original[0]), int(img_size_original[1])))
            try:
                if self.cfg['show_center']:
                    instant['center_heatmap'] = centers[0]
            except KeyError:
                pass
            kps_2d = np.reshape(pose_pre, [-1, 3])
            kps_hpecore = movenet_to_hpecore(kps_2d)
            kps_pre_hpecore = np.reshape(kps_hpecore[:,:2], [-1])
            kps_pre_hpecore_towrite = np.reshape(kps_hpecore[:,:2], [-1])
            # if kps_hpecore.shape[1]==3:
            instant['confidence'] = kps_hpecore[:,2]
            # print(kps_pre_hpecore)
            if write_csv is not None:
                # print('writing')
                row = self.create_row(ts,kps_pre_hpecore_towrite, delay=time.time()-start_sample)
                ensure_loc(os.path.dirname(write_csv))
                self.write_results(write_csv, row)

            instant['joints'] = kps_pre_hpecore
            # labels = list(hpecore_kps_labels.keys())
            # print('**********************')
            # for key in list(hpecore_kps_labels.keys()):
            #     print(key, kps_hpecore[hpecore_kps_labels[key],2])




            return instant


    def evaluate(self, data_loader,fastmode=False):
        self.model.eval()

        correct_kps = 0.0
        total_kps = 0.0
        joint_correct = np.zeros([self.cfg["num_classes"]])
        joint_total = np.zeros([self.cfg["num_classes"]])
        with torch.no_grad():
            start = time.time()
            for batch_idx, (imgs, labels, kps_mask, img_names, torso_diameter, head_size_norm, img_size_original, ts) in enumerate(data_loader):
                if img_size_original == 0:
                    continue

                start_sample = time.time()
                if batch_idx % 100 == 0 and batch_idx > 10:
                    print('Finished samples: ', batch_idx)
                    if not fastmode:
                        acc_intermediate = correct_kps / total_kps
                        acc_joint_mean_intermediate = np.mean(joint_correct / joint_total)
                        print('[Info] Mean Keypoint Acc: {:.3f}%'.format(100. * acc_intermediate))
                        print('[Info] Mean Joint Acc: {:.3f}%'.format(100. * acc_joint_mean_intermediate))
                        # print('Time since beginning:', time.time()-start)
                        print('[Info] Average Freq:', (batch_idx / (time.time() - start)), '\n')

                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs)

                pre = movenetDecode(output, kps_mask, mode='output', num_joints=self.cfg["num_classes"])
                gt = movenetDecode(labels, kps_mask, mode='label', num_joints=self.cfg["num_classes"])

                if not fastmode:
                    if torso_diameter is None:
                        pck_acc = pck(pre, gt, head_size_norm, num_classes=self.cfg["num_classes"], mode='head')
                    else:
                        pck_acc = pck(pre, gt, torso_diameter, threshold=0.5, num_classes=self.cfg["num_classes"],
                                      mode='torso')
                    correct_kps += pck_acc["total_correct"]
                    total_kps += pck_acc["total_keypoints"]
                    joint_correct += pck_acc["correct_per_joint"]
                    joint_total += pck_acc["anno_keypoints_per_joint"]

                    img_out, pose_gt = restore_sizes(imgs[0], gt, (int(img_size_original[0]), int(img_size_original[1])))
                    # print('gt after restore function', pose_gt)

                _, pose_pre = restore_sizes(imgs[0], pre, (int(img_size_original[0]), int(img_size_original[1])))
                # print('pre after restore function',pose_pre)

                kps_2d = np.reshape(pose_pre, [-1, 2])
                kps_hpecore = movenet_to_hpecore(kps_2d)
                kps_pre_hpecore = np.reshape(kps_hpecore, [-1])
                if self.cfg['write_output']:
                    row = self.create_row(ts,kps_pre_hpecore, delay=time.time()-start_sample)
                    sample = '_'.join(os.path.basename(img_names[0]).split('_')[:-1])
                    write_path = os.path.join(self.cfg['results_path'],self.cfg['dataset'],sample,'movenet.csv')
                    ensure_loc(os.path.dirname(write_path))
                    self.write_results(write_path, row)

        if not fastmode:
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
            for batch_idx, (imgs, labels, kps_mask, img_names, torso_diameter, head_size_norm, _, _) in enumerate(
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
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print('The checkpoint expected does not exist. Please check the path and filename.')

        if data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def write_results(self, path, row):
        # Write a data point into a csvfile
        with open(path, 'a') as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerow(row)

    def create_row(self, ts, skt, delay = 0.0):
        # Function to create a row to be written into a csv file.
        row = []
        ts = float(ts)
        row.extend([ts, delay])
        row.extend(skt)
        return row
