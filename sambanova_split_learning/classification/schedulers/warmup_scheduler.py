import math
from typing import List


class WarmupScheduler(object):
    def __init__(self, base_lr, iter_per_epoch, max_epoch, multi_step=[30, 60, 90], gamma=.1, warmup_epoch=5):
        super(WarmupScheduler, self).__init__()

        self.base_lr = base_lr
        self.warmup_iters = max(iter_per_epoch * warmup_epoch, 1)
        self.current_iter = 1
        self.last_lr = [base_lr / self.warmup_iters]
        if multi_step:
            self.mode = 'multi-step'
            self.get_lr = self.step_get_lr
            self.multi_step = multi_step
            self.iter_per_epoch = iter_per_epoch
            self.gamma = gamma
        else:
            self.mode = 'cosine'
            self.get_lr = self.cosine_get_lr
            self.cosine_iters = iter_per_epoch * (max_epoch - warmup_epoch)

    def __str__(self):
        res = 'Warmup Scheduler\n'
        res += f'Parameters:\n\tmode: {self.mode}\n\tbase_lr: {self.base_lr}\n\twarmup_iters: {self.warmup_iters}'
        if self.mode == 'multi-step':
            res += f'\n\tmulti_step: {self.multi_step}\n\tgamma: {self.gamma}\n\titer_per_epoch: {self.iter_per_epoch}\n'
        else:
            res += f'\n\tcosine_iters: {self.cosine_iters}\n'
        res += f'State:\n\tcurrent_iter: {self.current_iter}\n\tlast_lr: {self.last_lr}'
        return res

    def step_get_lr(self):
        if self.current_iter < self.warmup_iters:
            lr_ratio = self.current_iter / self.warmup_iters
        else:
            num_epochs = (self.current_iter - self.warmup_iters) / self.iter_per_epoch
            stage = sum([num_epochs > k for k in self.multi_step])
            lr_ratio = self.gamma**stage
        self.current_iter += 1
        return lr_ratio

    def cosine_get_lr(self):
        if self.current_iter < self.warmup_iters:
            lr_ratio = self.current_iter / self.warmup_iters
        else:
            process = (self.current_iter - self.warmup_iters) / self.cosine_iters
            lr_ratio = .5 * (1 + math.cos(process * math.pi))
            lr_ratio = max(lr_ratio, 1e-5)
        self.current_iter += 1
        return lr_ratio

    def get_last_lr(self) -> List[float]:
        '''
        Return learning rate per optimizer group, to mimic torch schedulers.
        '''
        return self.last_lr

    def step(self):
        lr_ratio = self.get_lr()
        self.last_lr = [lr_ratio * self.base_lr]
        return lr_ratio * self.base_lr
