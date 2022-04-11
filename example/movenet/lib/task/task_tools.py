"""
@Fire
https://github.com/fire717
"""

import torch.optim as optim
import numpy as np
import cv2

from lib.utils.utils import maxPoint, extract_keypoints

_range_weight_x = np.array([[x for x in range(48)] for _ in range(48)])
_range_weight_y = _range_weight_x.T


# _reg_weight = np.load("lib/data/my_weight_reg.npy") # 99x99


def getSchedu(schedu, optimizer):
    pass


def getOptimizer(optims, model, learning_rate, weight_decay):
    pass

############### Tools
def clipGradient(optimizer, grad_clip=1):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def movenetDecode(data, kps_mask=None, mode='output', num_joints=17,
                  img_size=192, hm_th=0.1):
    ##data [64, 7, 48, 48] [64, 1, 48, 48] [64, 14, 48, 48] [64, 14, 48, 48]
    # kps_mask [n, 7]

    if mode == 'output':
        batch_size = data[0].size(0)

        heatmaps = data[0].detach().cpu().numpy()

        heatmaps[heatmaps < hm_th] = 0

        centers = data[1].detach().cpu().numpy()

        regs = data[2].detach().cpu().numpy()
        offsets = data[3].detach().cpu().numpy()

        cx, cy = maxPoint(centers)

        dim0 = np.arange(batch_size, dtype=np.int32).reshape(batch_size, 1)
        dim1 = np.zeros((batch_size, 1), dtype=np.int32)

        res = []
        for n in range(num_joints):

            reg_x_origin = (regs[dim0, dim1 + n * 2, cy, cx] + 0.5).astype(np.int32)
            reg_y_origin = (regs[dim0, dim1 + n * 2 + 1, cy, cx] + 0.5).astype(np.int32)
            reg_x = reg_x_origin + cx
            reg_y = reg_y_origin + cy

            ### for post process
            reg_x = np.reshape(reg_x, (reg_x.shape[0], 1, 1))
            reg_y = np.reshape(reg_y, (reg_y.shape[0], 1, 1))
            reg_x = reg_x.repeat(48, 1).repeat(48, 2)
            reg_y = reg_y.repeat(48, 1).repeat(48, 2)

            range_weight_x = np.reshape(_range_weight_x, (1, 48, 48)).repeat(reg_x.shape[0], 0)
            range_weight_y = np.reshape(_range_weight_y, (1, 48, 48)).repeat(reg_x.shape[0], 0)
            tmp_reg_x = (range_weight_x - reg_x) ** 2
            tmp_reg_y = (range_weight_y - reg_y) ** 2

            tmp_reg = (tmp_reg_x + tmp_reg_y) ** 0.5 + 1.8  # origin 1.8

            tmp_reg = heatmaps[:, n, ...] / tmp_reg

            tmp_reg = tmp_reg[:, np.newaxis, :, :]
            reg_x, reg_y = maxPoint(tmp_reg, center=False)


            reg_x[reg_x > 47] = 47
            reg_x[reg_x < 0] = 0
            reg_y[reg_y > 47] = 47
            reg_y[reg_y < 0] = 0

            score = heatmaps[dim0, dim1 + n, reg_y, reg_x]
            offset_x = offsets[dim0, dim1 + n * 2, reg_y, reg_x]  # *img_size//4
            offset_y = offsets[dim0, dim1 + n * 2 + 1, reg_y, reg_x]  # *img_size//4
            res_x = (reg_x + offset_x) / (img_size // 4)
            res_y = (reg_y + offset_y) / (img_size // 4)

            res_x[score < hm_th] = -1
            res_y[score < hm_th] = -1

            res.extend([res_x, res_y])

        res = np.concatenate(res, axis=1)  # bs*14



    elif mode == 'label':
        kps_mask = kps_mask.detach().cpu().numpy()

        data = data.detach().cpu().numpy()
        batch_size = data.shape[0]

        heatmaps = data[:, :num_joints, :, :]
        centers = data[:, num_joints, :, :]
        regs = data[:, num_joints + 1:(num_joints * 3) + 1, :, :]
        offsets = data[:, (num_joints * 3) + 1:, :, :]

        cx, cy = maxPoint(centers)

        dim0 = np.arange(batch_size, dtype=np.int32).reshape(batch_size, 1)
        dim1 = np.zeros((batch_size, 1), dtype=np.int32)

        res = []
        for n in range(num_joints):

            reg_x_origin = (regs[dim0, dim1 + n * 2, cy, cx] + 0.5).astype(np.int32)
            reg_y_origin = (regs[dim0, dim1 + n * 2 + 1, cy, cx] + 0.5).astype(np.int32)


            reg_x = reg_x_origin + cx
            reg_y = reg_y_origin + cy

            reg_x[reg_x > 47] = 47
            reg_x[reg_x < 0] = 0
            reg_y[reg_y > 47] = 47
            reg_y[reg_y < 0] = 0

            offset_x = offsets[dim0, dim1 + n * 2, reg_y, reg_x]  # *img_size//4
            offset_y = offsets[dim0, dim1 + n * 2 + 1, reg_y, reg_x]  # *img_size//4
            res_x = (reg_x + offset_x) / (img_size // 4)
            res_y = (reg_y + offset_y) / (img_size // 4)


            res_x[kps_mask[:, n] == 0] = -1
            res_y[kps_mask[:, n] == 0] = -1
            res.extend([res_x, res_y])

        res = np.concatenate(res, axis=1)  # bs*14

    return res

def restore_sizes(img,pose,size_out):
    size_in = img.shape
    img_out = cv2.resize(img,(size_out[1],size_out[0]))
    pose_out = np.copy(pose.reshape((-1,2)))
    for i in range(len(pose_out)):
        pose_out[i,0] = pose_out[i,0] * size_out[1]
        pose_out[i,1] = pose_out[i,1] * size_out[0]
    return img_out, np.round(pose_out)


def image_show(img,pre=None):

    # img = np.transpose(img[0].cpu().numpy(), axes=[1, 2, 0])
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if np.amax(img) >1:
        img = img/255

    if pre is not None:
        pre[pre[:]<0]=0

        if len(pre.squeeze().shape) == 1:
            for i in range(len(pre[0]) // 2):
                x = int(pre[0][i * 2] * w)
                y = int(pre[0][i * 2 + 1] * h)
                cv2.circle(img, (x, y), 3, (255, 0, 0), 2)
        else:
            for i in range(len(pre)):
                cv2.circle(img, (int(pre[i,0]), int(pre[i,1])), 3, (255, 0, 0), 2)

    img = cv2.resize(img,(img.shape[0]*4,img.shape[1]*4))
    cv2.imshow("output", img)


    return 0
