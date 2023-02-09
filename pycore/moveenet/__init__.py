"""
@Fire
https://github.com/fire717
"""
import os

from pycore.moveenet.data.data import Data
from pycore.moveenet.models.movenet_mobilenetv2 import MoveNet
from pycore.moveenet.task.task import Task


from pycore.moveenet.utils.utils import setRandomSeed, printDash


def init(cfg):

    if cfg["cfg_verbose"]:
        printDash()
        print(cfg)
        printDash()



    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['GPU_ID']
    setRandomSeed(cfg['random_seed'])


    #
    # if not os.path.exists(cfg['save_dir']):
    #     os.makedirs(cfg['save_dir'])