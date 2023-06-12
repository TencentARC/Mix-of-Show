import numpy as np
import torch
from copy import deepcopy
from torch import nn as nn
from torch.nn import functional as F

from mixofshow.utils import get_root_logger
from mixofshow.utils.misc import inmap_func
from mixofshow.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


def build_loss(opt):
    """Build loss from options.
    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.
    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class KDELoss(nn.Module):

    def __init__(self, loss_weight=0.002, bandwidth=0.5):
        super(KDELoss, self).__init__()
        self.bandwidth = bandwidth
        self.loss_weight = loss_weight

    def gaussian_kernel(self, u):
        return (1 / (torch.sqrt(2 * torch.tensor(np.pi)) * self.bandwidth)) * torch.exp(-0.5 * (u / self.bandwidth)**2)

    def forward(self, embeddings, x_batch):
        batch_size = x_batch.size(0)

        total_log_density = 0

        for i in range(batch_size):
            differences = x_batch[i] - embeddings
            kernel_values = self.gaussian_kernel(differences)
            mean_kernel_values = torch.mean(kernel_values, dim=0)
            density = torch.sum(mean_kernel_values, dim=0)
            total_log_density += -torch.log(density)

        return self.loss_weight * total_log_density


@LOSS_REGISTRY.register()
class CrossEntropyLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', label_smooth=0.0, ignore_index=-1):
        super(CrossEntropyLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}.')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.label_smooth = label_smooth
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return F.cross_entropy(
            input, target, reduction=self.reduction, label_smoothing=self.label_smooth, ignore_index=self.ignore_index)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.
    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class LogitLaplaceLoss(nn.Module):
    """MSE (L2) loss.
    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(LogitLaplaceLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def cal_logit_laplace(self, pred, target):
        pred_map = inmap_func(pred)
        target_map = inmap_func(target)
        loss = torch.log(1 / (2 * target_map *
                              (1 - target_map))) - torch.abs(torch.logit(pred_map) - torch.logit(target_map))
        return -loss.mean()

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """

        return self.loss_weight * self.cal_logit_laplace(pred, target)
