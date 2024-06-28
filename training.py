import argparse
from processor import Processor
import numpy as np
import torch
import random
import yaml


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Graph Convolution Network')

    parser.add_argument(
        '--config',
        type=str,
        default=r'D:\DATN\project\Pose-based-WLASL\configs\ctr-gcn\config.yaml',
        help='Path to the configuration file')

    parser.add_argument(
        '--weight',
        default=None,
        help='The weights for network initialization')

    return parser


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    config_path = p.config
    with open(config_path, 'r') as f:
        arg = yaml.load(f, yaml.FullLoader)
    arg = argparse.Namespace(**arg)
    arg.weight = p.weight
    # Read config
    init_seed(0)
    processor = Processor(arg)
    processor.start()
