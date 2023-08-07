import torch
import warnings

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)

# --------------------- Helper Class --------------------- # 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Scheduler_Helper(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False, module_name = ""):

        super().__init__(optimizer, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps, verbose)
        self.module_name = module_name

        self.n_steps = 0

    def _reduce_lr(self, epoch):
        msgs = ""

        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                                "%.5d") % epoch
                # print('Epoch {}: reducing learning rate'
                #       ' of {}`s group {} to {:.4e}.'.format(epoch_str, self.module_name, i, new_lr))
                msgs += f'Epoch {epoch_str}: reducing learning rate of {self.module_name}`s group {i} to {new_lr:.4e}.\n'
        if self.verbose:
            print(msgs)

        return msgs
    

    def step(self, metrics, epoch=None):
        # metric 개선 여부와 상관 없이 step 수 계산. 
        # 이 값이 25이상이면 강제로 한번 내리고 초기화 
        self.n_steps += 1

        # convert `metrics` to float, in case it's a zero-dim Tensor
        msgs = None
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            msgs = self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            self.n_steps = 0

        if self.n_steps > 25:
            msgs = self._reduce_lr(epoch)
            self.n_steps = 0 

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

        return msgs


