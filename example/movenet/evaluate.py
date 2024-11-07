"""
@Fire
https://github.com/fire717
"""
import os, argparse
import random
import sys

sys.path.append('.')
sys.path.append('../../../hpe-core')
from pycore.moveenet import init, Data, MoveNet, Task

from config import cfg
from pycore.moveenet.utils.utils import arg_parser


def main(cfg):
    init(cfg)

    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')

    data = Data(cfg)
    data_loader = data.getEvalDataloader()

    run_task = Task(cfg, model)

    run_task.modelLoad("models/e97_valacc0.81209.pth")

    run_task.evaluate(data_loader, fastmode=True)
    # run_task.infer_video(data_loader,'/home/ggoyal/data/h36m/tester/out.avi')


if __name__ == '__main__':
    cfg = arg_parser(cfg)
    main(cfg)
