import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch_optimizer as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.optim.lr_scheduler import _LRScheduler


def set_data_split(args, train_data, train_labels):
    folds = []
    if args.data_split.lower() == 'split_base':
        train_data, val_data, train_lb, val_lb = \
            train_test_split(train_data, train_labels, test_size=args.val_ratio, random_state=args.seed, shuffle=True)
        folds.append((train_data, train_lb, val_data, val_lb))
    elif args.data_split.lower() == 'stratifiedkfold':
        train_data = np.array(train_data)
        skf = StratifiedKFold(n_splits=args.n_splits, random_state=args.seed, shuffle=True)
        for train_idx, valid_idx in skf.split(train_data, train_labels):
            train_labels = np.array(train_labels)
            folds.append((train_data[train_idx].tolist(), train_labels[train_idx].tolist(),
                          train_data[valid_idx].tolist(), train_labels[valid_idx].tolist()))
    else:
        pass
    return folds


def set_optimizer(args, model):
    optimizer = None
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
    elif args.optimizer == 'radam':
        optimizer = optim.RAdam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'adamp':
        optimizer = optim.AdamP(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'ranger':
        optimizer = optim.Ranger(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay)
    elif args.optimizer == 'lamb':
        optimizer = optim.Lamb(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    return optimizer


def set_loss(args):
    criterion = None
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'focal':
        criterion = FocalLoss()

    return criterion


def set_scheduler(args, optimizer, iter_per_epoch):
    scheduler = None
    if args.scheduler == 'cos_base':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'cos':
        # tmax = epoch * 2 => half-cycle
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=args.min_lr)
    elif args.scheduler == 'cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr,
                                                        steps_per_epoch=iter_per_epoch, epochs=args.epochs)

    return scheduler


def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, factor=0.1, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights
        self.factor = factor

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets


def score_function(real, pred):
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = accuracy_score(real, pred)
    return score


def init_logger(save_dir, comment=None):
    c_date, c_time = datetime.now().strftime("%Y%m%d/%H%M%S").split('/')
    if comment is not None:
        if os.path.exists(os.path.join(save_dir, c_date, comment)):
            comment += f'_{c_time}'
    else:
        comment = c_time
    log_dir = os.path.join(save_dir, c_date, comment)

    return log_dir
