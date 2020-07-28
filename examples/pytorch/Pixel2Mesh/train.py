import sys
import os
import argparse

import numpy as np
import torch

from mesh_utils import Ellipsoid
from dataset import ShapeNetDataset, get_shapenet_collate
from models import Pixel2MeshModel

def parse_args():
    parser = argparse.ArgumentParser(description='Pixel2Mesh Training')
    #parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    # training
    parser.add_argument('--batch-size', help='batch size', type=int, default=32)
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument('--dataset-dir', help='dataset directory', type=str)
    parser.add_argument('--ckpt-dir', help='checkpoint directory', type=str)
    parser.add_argument('--num-workers', help='number of workers', type=int, default=0)
    parser.add_argument('--num-epochs', help='number of epochs', type=int)

    # model param
    parser.add_argument('--hidden-dim', help='gcn hidden layer dimension', type=int, default=192)
    parser.add_argument('--last-hidden-dim', help='last gcn hidden layer dimension', type=int, default=192)
    parser.add_argument('--coord-dim', help='coordinate dimension', type=int, default=3)
    parser.add_argument('--pretrained-backbone', help='pretrained backbone path', type=str, default=None)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # create dataset
    train_file_list_path = os.path.join(args.dataset_dir, 'meta/train_tf.txt')
    train_set = ShapeNetDataset(args.dataset_dir, train_file_list_path)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               collate_fn=get_shapenet_collate,
                                               shuffle=True)
    test_file_list_path = os.path.join(args.dataset_dir, 'meta/test_tf.txt')
    test_set = ShapeNetDataset(args.dataset_dir, test_file_list_path)
    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               collate_fn=get_shapenet_collate,
                                               shuffle=False)

    # Create model
    mesh_file = os.path.join(args.dataset_dir, '../ellipsoid/info_ellipsoid.dat')
    ellipsoid = Ellipsoid(mesh_file)

    model = Pixel2MeshModel(args.hidden_dim, args.last_hidden_dim, args.coord_dim, ellipsoid, pretrained_backbone=args.pretrained_backbone)

    print (model)

    exit (0)

    # Create loss
    loss = P2MLoss(option, eclipse)

    # Training Loop
    for epoch in range(args.num_epochs):
        tic = time.time()
        loss_val = 0
        for data in train_loader:
            opt.zero_grad()
            out = model(data)
            loss = loss(out, data)
            loss.backward()
            opt.step()
            scheduler.step()
            time_diff = time.time() - tic
            print('Epoch #%d, loss=%f, time=%.6f' % (epoch, loss.sum(), time_diff))
            if (epoch + 1) % args.model_save_freq == 0:
                torch.save(net.state_dict(), model_filename)
        # eval
        for data in train_loader:
            opt.zero_grad()
            out = model(data)
            loss = loss(out, data)
            loss.backward()
            opt.step()
            scheduler.step()
            time_diff = time.time() - tic
            print('Epoch #%d, loss=%f, time=%.6f' % (epoch, loss.sum(), time_diff))
            if (epoch + 1) % args.model_save_freq == 0:
                torch.save(net.state_dict(), model_filename)

if __name__ == "__main__":
    np.random.seed(1234)
    torch.manual_seed(1234)
    main()
