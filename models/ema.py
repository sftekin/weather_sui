import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EMA(nn.Module):

    def __init__(self, constant_params, finetune_params):
        super(EMA, self).__init__()
        self.window_len = constant_params['window_len']
        self.mu = finetune_params.get('mu', 2 / (self.window_len + 1))

    def fit(self, X, y, **kwargs):
        """
        :param X: Tensor, (b, t, m, n, d)
        :param y: Tensor, (b, 1, m, n, 1)
        :param kwargs:
        :return: loss numpy
        """
        X, y = X.to(device), y.to(device)
        X = X.permute(0, 1, 4, 2, 3)
        y = y.permute(0, 1, 4, 2, 3)

        # initial value equal to mean of the batch
        last_average = torch.mean(X, dim=1, keepdim=True)
        for t in range(1, self.window_len):
            last_average = self.forward(X[:, t-1], last_average)

        loss = self.compute_loss(y, last_average)
        return loss.detach().cpu().numpy()

    def predict(self, X):
        """
        :param X: Tensor, (b, t, m, n, d)
        :return: numpy array, b, t, m, n, d)
        """
        X = X.to(device)
        X = X.permute(0, 1, 4, 2, 3)

        # initial value equal to mean of the batch
        pred = torch.mean(X, dim=1, keepdim=True)
        for t in range(1, self.window_len):
            pred = self.forward(X[:, t-1], pred)
        return pred.detach().cpu().numpy()

    def forward(self, x, last_average):
        new_average = self.mu * x + (1 - self.mu) * last_average
        return new_average

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
