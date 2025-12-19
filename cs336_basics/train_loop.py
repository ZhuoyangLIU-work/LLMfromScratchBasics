import argparse
import time
import torch
import numpy as np
import typing, os
from cs336_basics.train import *


import logging
ts = time.strftime("%Y%m%d_%H%M%S")
logging.basicConfig(filename=f"train_loop_start_{ts}.log", filemode='w', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer from tokens")
    # parser arguments

    # check pointing
    parser.add_argument("-cint", "--checkpoint-interval", required=True, help="model training check point interval")

    # data loading
    parser.add_argument("-bs", "--batch-size", required=True, help="batch size")
    parser.add_argument("-cl", "--context-length", required=True, help="context length")

    return parser

def train_from_ndafile(config, from_checkpoint = False):
    src = config["src"]
    dataset_memmap = load_data_from_file(src)

    







