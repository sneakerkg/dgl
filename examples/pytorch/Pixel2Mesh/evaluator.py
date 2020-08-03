from logging import Logger
from collections import Iterable

import numpy as np
import torch
import torch.nn as nn

from chamfer_wrapper import ChamferDist
from mesh_utils import Ellipsoid
#from utils.vis.renderer import MeshRenderer

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.cpu().numpy()
        if isinstance(val, Iterable):
            val = np.array(val)
            self.update(np.mean(np.array(val)), n=val.size)
        else:
            self.val = self.multiplier * val
            self.sum += self.multiplier * val * n
            self.count += n
            self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self):
        return "%.6f (%.6f)" % (self.val, self.avg)

class Evaluator(object):

    def __init__(self, num_classes):
        # Evaluate step count, useful in summary
        self.evaluate_step_count = 0
        self.total_step_count = 0
        # Creating meters
        self.num_classes = num_classes
        self.chamfer_distance = [AverageMeter() for _ in range(self.num_classes)]
        self.f1_tau = [AverageMeter() for _ in range(self.num_classes)]
        self.f1_2tau = [AverageMeter() for _ in range(self.num_classes)]

    def reset_eval_step(self):
        self.evaluate_step_count = 0
        self.chamfer_distance = [AverageMeter() for _ in range(self.num_classes)]
        self.f1_tau = [AverageMeter() for _ in range(self.num_classes)]
        self.f1_2tau = [AverageMeter() for _ in range(self.num_classes)]

    def average_of_average_meters(self, average_meters):
        s = sum([meter.sum for meter in average_meters])
        c = sum([meter.count for meter in average_meters])
        weighted_avg = s / c if c > 0 else 0.
        avg = sum([meter.avg for meter in average_meters]) / len(average_meters)
        ret = AverageMeter()
        if self.weighted_mean:
            ret.val, ret.avg = avg, weighted_avg
        else:
            ret.val, ret.avg = weighted_avg, avg
        return ret

    def get_result_summary(self):
        return {
            "cd": self.average_of_average_meters(self.chamfer_distance),
            "f1_tau": self.average_of_average_meters(self.f1_tau),
            "f1_2tau": self.average_of_average_meters(self.f1_2tau),
        }

    def evaluate_f1(self, dis_to_pred, dis_to_gt, pred_length, gt_length, thresh):
        recall = np.sum(dis_to_gt < thresh) / gt_length
        prec = np.sum(dis_to_pred < thresh) / pred_length
        return 2 * prec * recall / (prec + recall + 1e-8)

    def evaluate_chamfer_and_f1(self, pred_vertices, gt_points, labels):
        # calculate accurate chamfer distance; ground truth points with different lengths;
        # therefore cannot be batched
        batch_size = pred_vertices.size(0)
        pred_length = pred_vertices.size(1)
        for i in range(batch_size):
            gt_length = gt_points[i].size(0)
            label = labels[i].cpu().item()
            d1, d2, i1, i2 = self.chamfer(pred_vertices[i].unsqueeze(0), gt_points[i].unsqueeze(0))
            d1, d2 = d1.cpu().numpy(), d2.cpu().numpy()  # convert to millimeter
            self.chamfer_distance[label].update(np.mean(d1) + np.mean(d2))
            self.f1_tau[label].update(self.evaluate_f1(d1, d2, pred_length, gt_length, 1E-4))
            self.f1_2tau[label].update(self.evaluate_f1(d1, d2, pred_length, gt_length, 2E-4))
