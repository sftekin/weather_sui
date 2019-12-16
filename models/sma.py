import torch
import torch.nn as nn

from scipy import signal

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
        # TODO add left right middle attention
        if self.init_dist == 'gaussian':
            weight = None
        elif self.init_dist == 'kaiser':
            weight = None
        else:
            uniform = None



