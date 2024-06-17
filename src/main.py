from utils.parser import parse_args
from utils.util import *
from utils.data_loader import Dataloader
from train import train
import logging

if __name__ == '__main__':
    """read args"""
    args = parse_args()
    print(args.seed)
    """set seed"""
    set_seed(args.seed)

    """set log"""
    set_log(args)

    """load data"""
    data = Dataloader(args, logging)

    """start training"""
    train(args, data)
