import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EMA(nn.Module):

    def __init__(self, constant_params, finetune_params):
        super(EMA, self).__init__()
        self.window_len = constant_params['window_len']
        self.label_feature = constant_params['label_feature']
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

        pred = self.forward(X)

        loss = self.compute_loss(y, pred)
        return loss.detach().cpu().numpy()

    def predict(self, X):
        """
        :param X: Tensor, (b, t, m, n, d)
        :return: numpy array, b, t, m, n, d)
        """
        X = X.to(device)
        X = X.permute(0, 1, 4, 2, 3)

        # initial value equal to mean of the batch
        pred = self.forward(X)
        return pred.detach().cpu().numpy()

    def forward(self, x):
        """
        :param x: 5-D tensor of shape (b, t, m, n, d)
        :return: (b, t, m, n, d)
        """
        # first trace the given data
        input_len = x.shape[1]
        # initial value equal to mean of the batch
        last_average = torch.mean(x, dim=1, keepdim=True)
        for t in range(1, input_len+1):
            last_average = self.mu * x[:, [t - 1]] + (1 - self.mu) * last_average

        # second recurrently make predictions until output window length
        output = []
        # initial value equal to last element of the batch
        t_minus_one = x[:, [-1]]
        for t in range(self.window_len):
            output.append(last_average)
            last_average = self.mu * t_minus_one + (1 - self.mu) * last_average
            t_minus_one = output[t]

        output = torch.cat(output, dim=1)

        return output[:, :, self.label_feature]

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
