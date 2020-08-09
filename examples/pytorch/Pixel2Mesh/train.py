import sys
import os
import time
import argparse

import numpy as np
import torch
from tensorboardX import SummaryWriter

from losses import P2MLoss
from evaluator import Evaluator
from mesh_utils import Ellipsoid
from models import Pixel2MeshModel
from utils import create_logger
from dataset import ShapeNetDataset, get_shapenet_collate


def parse_args():
    parser = argparse.ArgumentParser(description='Pixel2Mesh Training')
    #parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    # training
    parser.add_argument('--batch-size', help='batch size', type=int, default=32)
    parser.add_argument('--dataset-dir', help='dataset directory', type=str)
    parser.add_argument('--num-workers', help='number of workers', type=int, default=0)
    parser.add_argument('--num-epochs', help='number of epochs', type=int, default=100)

    # dataset param
    parser.add_argument('--num-points', help='num points to sample from shapenet data', type=int, default=9000)

    # model param
    parser.add_argument('--hidden-dim', help='gcn hidden layer dimension', type=int, default=192)
    parser.add_argument('--last-hidden-dim', help='last gcn hidden layer dimension', type=int, default=192)
    parser.add_argument('--coord-dim', help='coordinate dimension', type=int, default=3)
    parser.add_argument('--pretrained-backbone', help='pretrained backbone path', type=str, default=None)

    # log and eval
    parser.add_argument('--log-dir', help='checkpoint directory', type=str)
    parser.add_argument('--ckpt-freq', help='checkpoint frequency', type=int, default=10)
    parser.add_argument('--train-summary-freq', help='train summary frequency', type=int, default=100)
    parser.add_argument('--test-summary-freq', help='test summary frequency', type=int, default=100)

    args = parser.parse_args()
    return args

def write_summary(pred, gt, loss_summary, logger, writer, phase, epoch, epoch_iter, global_iter_num):
        # Save results in Tensorboard
        for k, v in loss_summary.items():
            writer.add_scalar("%s, %s"%(phase, k), v, global_iter_num)

        # Save results to log
        log_info = "%s Epoch %d, Step %d "%(phase, epoch, epoch_iter)
        for k, v in loss_summary.items():
            log_info += "%s %f, " % (k, v)
        self.logger.info(log_info)

def write_result_summary(logger, writter, evaluator):
    for key, val in evaluator.get_result_summary().items():
        scalar = val
        if isinstance(val, AverageMeter):
            scalar = val.avg
        self.logger.info("Test [%06d] %s: %.6f" % (self.total_step_count, key, scalar))
        self.summary_writer.add_scalar("eval_" + key, scalar, self.total_step_count + 1)

def main():
    args = parse_args()

    # create dataset
    train_file_list_path = os.path.join(args.dataset_dir, 'meta/train_tf.txt')
    train_set = ShapeNetDataset(args.dataset_dir, train_file_list_path)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               collate_fn=get_shapenet_collate(args.num_points),
                                               shuffle=True)
    test_file_list_path = os.path.join(args.dataset_dir, 'meta/test_tf.txt')
    test_set = ShapeNetDataset(args.dataset_dir, test_file_list_path)
    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               collate_fn=get_shapenet_collate(args.num_points),
                                               shuffle=False)

    # Create model
    mesh_file = os.path.join(args.dataset_dir, '../ellipsoid/info_ellipsoid.dat')
    ellipsoid = Ellipsoid(mesh_file)

    model = Pixel2MeshModel(args.hidden_dim, args.last_hidden_dim, args.coord_dim, ellipsoid, pretrained_backbone=args.pretrained_backbone)

    # Create loss
    loss = P2MLoss(ellipsoid)

    # Params should be put into yml
    adam_beta1 = 0.9
    lr = 0.0001
    lr_factor = 0.3
    lr_step = [30, 70, 90]
    name = 'adam'
    sgd_momentum = 0.9
    wd = 1.0e-06

    # Create optimizer
    optimizer = torch.optim.Adam(
        params=list(model.parameters()),
        lr=lr,
        betas=(adam_beta1, 0.999),
        weight_decay=wd
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, lr_step, lr_factor
    )

    # Create evaluator
    num_class = len(train_set.id_name_map)
    evaluator = Evaluator(num_class)

    # Create logger and writer
    os.makedirs(args.log_dir, exist_ok=True)
    logger = create_logger(args.log_dir, log_level='info')
    writer = SummaryWriter(args.log_dir)

    # Training Loop
    global_iter_num = 0
    for epoch in range(args.num_epochs):
        tic = time.time()
        loss_val = 0
        for step, data in enumerate(train_loader):
            optimizer.zero_grad()

            print (data)

            out = model(data)

            print (out)

            loss, loss_summary = loss(out, data)

            print (loss)

            loss.backward()
            optimizer.step()
            scheduler.step()

            global_iter_num += 1
            if (step + 1) % args.train_summary_freq == 0:
                write_summary(out, data, loss_summary, logger, writer, 'Train', epoch, step, global_iter_num)

        if (epoch + 1) % args.ckpt_freq == 0:
            torch.save(net.state_dict(), model_filename)
        time_diff = time.time() - tic

        # eval
        evaluator = Evaluator(num_class)
        model.eval()
        with torch.no_grad():
            for step, data in enumerate(test_loader):
                out = model(data)
                loss, loss_summary = loss(out, data)
                evaluator.evaluate_chamfer_and_f1(out, data, data["labels"])
                if (step + 1) % args.test_summary_freq == 0:
                    write_summary(out, data, loss_summary, logger, writer, 'Test', epoch, step, global_iter_num)
        write_result_summary(logger, writter, evaluator)
        model.train()


if __name__ == "__main__":
    np.random.seed(1234)
    torch.manual_seed(1234)
    main()
