"""
@Fire
https://github.com/fire717
"""

from lib import init, Data, MoveNet, Task

from config import cfg
from lib.utils.utils import arg_parser


# Script to create and save as images all the various outputs of the model


def main(cfg):
    init(cfg)

    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')

    data = Data(cfg)
    test_loader = data.getTestDataloader()

    run_task = Task(cfg, model)
    run_task.modelLoad("/output/mpii_pre-trained.pth")

    run_task.predict(test_loader, "/output/predict")

if __name__ == '__main__':
    cfg = arg_parser(cfg)
    main(cfg)
