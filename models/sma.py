import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from scipy import signal


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SMA(nn.Module):

    def __init__(self, constant_params, finetune_params):
        super(SMA, self).__init__()
        self.window_len = constant_params['window_len']
        self.output_len = constant_params['output_len']
        self.init_dist = constant_params.get('init_dist', 'gaussian')
        self.train_weights = constant_params['train_weights']
        self.attention_to = constant_params['attention_to']
        self.weight = self.__init_weight()

        if self.train_weights:
            self.lr = finetune_params['lr']
            self.set_optimizer()

    def fit(self, X, y, **kwargs):
        """
        :param X: (b, t, m, n, d)
        :param y: (b, 1, m, n, 1)
        :return: loss numpy
        """
        X, y = X.to(device), y.to(device)
        X = X.permute(0, 1, 4, 2, 3)
        y = y.permute(0, 1, 4, 2, 3)

        pred = self.forward(X, **kwargs)

        if self.train_weights:
            self.optimizer.zero_grad()
        loss = self.compute_loss(y, pred)

        if self.train_weights:
            loss.backward()
            self.optimizer.step()

        return loss.detach().cpu().numpy()

    def predict(self, X):
        X = X.to(device)
        X = X.permute(0, 1, 4, 2, 3)

        pred = self.forward(X)
        return pred.detach().cpu().numpy()

    def forward(self, input_tensor, **kwargs):
        """
        :param input_tensor: 5-D tensor of shape (b, t, m, n, d)
        :return: (b, t, m, n, d)
        """
        output = []
        for i in range(self.output_len):
            avg = self._predict_window(input_tensor)
            output.append(avg)
            input_tensor = torch.cat([input_tensor[:, 1:], avg], dim=1)

        output = torch.cat(output, dim=1)
        return output

    def _predict_window(self, x):
        for t in range(self.window_len):
            x[:, t] *= self.weight[t]

        # average on time dimension
        pred = torch.mean(x, dim=1, keepdim=True)
        return pred

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def score(self, labels, preds):
        """
        Given labels and preds arrays, compute loss
        :param labels: np array
        :param preds: np array
        :return: loss (float)
        """
        loss = self.compute_loss(torch.Tensor(labels), torch.Tensor(preds))
        return loss.numpy()

    @staticmethod
    def compute_loss(labels, preds):
        criterion = nn.MSELoss()
        loss = criterion(preds, labels)
        return loss.mean()

    def __init_weight(self):
        """
        :return: (window_len,) numpy.array
        """
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
            window = window[self.window_len//4: self.window_len//4 + self.window_len]

        window = torch.from_numpy(window).float()
        if self.train_weights:
            window = nn.Parameter(window).to(device)
        return window


