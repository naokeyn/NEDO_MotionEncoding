import os
import sys
import json

from argparse import ArgumentParser
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.utils import *
from src.model import ResNet1d, CNN2LSTM
from src.preprocess import EmgEncoder


def get_args():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="path to config file")
    
    args = parser.parse_args()
    
    return args


def main():
    args = get_args()
    config_path = args.config
    params = load_json(config_path)
    
    print(params)
    return

if __name__ == "__main__":
    main()
    