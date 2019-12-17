import torch
import numpy as np
import torch.nn as nn

from scipy import signal
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SMA(nn.Module):

    def __init__(self, constant_params, finetune_params):
        super(SMA, self).__init__()
        self.window_len = finetune_params['window_len']
        self.init_dist = constant_params.get('init_dist', 'gaussian')
        self.train_weight = constant_params['train_weight']
        self.attention_to = constant_params['right']
        self.weight = self.__init_weight()

    def __init_weight(self):
        filter_len = 2 * self.window_len
        if self.init_dist == 'gaussian':
            window = signal.gaussian(filter_len, std=10)
        elif self.init_dist == 'kaiser':
            window = signal.kaiser(filter_len, beta=14)
        else:
            # uniform
            window = np.ones(filter_len)

        if self.attention_to == 'right':
            window = window[:self.window_len]
        elif self.attention_to == 'left':
            window = window[self.window_len:]
        else:
            # middle
            window = window[self.window_len//4:self.window_len]

        window = torch.from_numpy(window)
        if self.train_weight:
            window = Variable(window).to(device)
        return window


