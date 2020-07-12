import sys
import os
import argparse

import numpy as np
import torch

from dataset import ShapeNetDataset, get_shapenet_collate
from functions.trainer import Trainer
from options import update_options, options, reset_options


def parse_args():
    parser = argparse.ArgumentParser(description='Pixel2Mesh Training Entrypoint')
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    # training
    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument('--dataset-dir', help='dataset directory', type=str)
    parser.add_argument('--ckpt-dir', help='checkpoint directory', type=str)
    parser.add_argument('--num-workers', help='number of workers', type=int)
    parser.add_argument('--num-epochs', help='number of epochs', type=int)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # create dataset
    train_file_list_path = os.path.join(args.dataset_dir, '/meta/train_tf.txt')
    train_set = ShapeNetDataset(args.dataset_dir, train_file_list_path)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               collate_fn=get_shapenet_collate,
                                               shuffle=True)
    test_file_list_path = os.path.join(args.dataset_dir, '/meta/test_tf.txt')
    test_set = ShapeNetDataset(args.dataset_dir, test_file_list_path)
    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               collate_fn=get_shapenet_collate,
                                               shuffle=False)

    # create model




    logger, writer = reset_options(options, args)

    trainer = Trainer(options, logger, writer)
    trainer.train()


if __name__ == "__main__":
    np.random.seed(1234)
    torch.manual_seed(1234)
    parser = argparse.ArgumentParser(description='Pixel2Mesh DGL Training')
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    # training
    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument('--num-epochs', help='number of epochs', type=int)
    parser.add_argument('--version', help='version of task (timestamp by default)', type=str)
    parser.add_argument('--name', required=True, type=str)

    args = parser.parse_args()


    main()
