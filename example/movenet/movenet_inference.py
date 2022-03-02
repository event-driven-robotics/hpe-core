"""
@Fire
https://github.com/fire717
"""
import os, argparse
import random
import sys
import gc
import torch
import numpy as np
import cv2
import time

sys.path.append('.')
from lib import init, Data, MoveNet, Task

from config import cfg
from lib.utils.utils import arg_parser


def main(cfg):
    init(cfg)

    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')

    data = Data(cfg)
    data_loader = data.getEvalDataloader()

    run_task = Task(cfg, model)

    # run_task.modelLoad("output/mpii_pre-trained.pth")
    run_task.modelLoad("/home/ggoyal/data/h36m/output/h36m_finetune_sub/best.pth")

    # run_task.evaluate(data_loader)
    run_task.infer_video(data_loader,'/home/ggoyal/data/h36m/tester/out.avi')

    # Initializations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # edit for Franklin
    model = model.to(device)

    ############################################################


    model.eval()

    correct_kps = 0.0
    total_kps = 0.0
    size = cfg["img_size"]

    with torch.no_grad():


            img = img.to(device)
            kps_mask = kps_mask.to(device)

            output = model(imgs)

            pre = movenetDecode(output, kps_mask, mode='output', num_joints=cfg["num_classes"])
            gt = movenetDecode(labels, kps_mask, mode='label', num_joints=cfg["num_classes"])

            if cfg['dataset'] in ['coco', 'mpii']:
                pck_acc = pck(pre, gt, head_size_norm, num_classes=cfg["num_classes"], mode='head')
                th_val = head_size_norm
            else:
                pck_acc = pck(pre, gt, torso_diameter, num_classes=cfg["num_classes"], mode='torso')
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


if __name__ == '__main__':
    cfg = arg_parser(cfg)
    main(cfg)
