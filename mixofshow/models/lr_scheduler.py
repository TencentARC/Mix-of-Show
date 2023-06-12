from torch.optim.lr_scheduler import _LRScheduler


class LinearLR(_LRScheduler):

    def __init__(self, optimizer, num_epochs, last_epoch=-1):
        self.num_epochs = num_epochs
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.num_epochs) for base_lr in self.base_lrs]


class ConstantLR(_LRScheduler):

    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.base_lrs
