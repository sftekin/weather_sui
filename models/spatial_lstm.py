import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SpatialLSTM(nn.Module):
    def __init__(self, constant_params, finetune_params):
        super(SpatialLSTM, self).__init__()
        self.height, self.width = constant_params['input_size']
        self.seq_len = constant_params['seq_len']
        self.input_size = self.height * self.width

        # lstm conf
        self.input_dim = constant_params['input_dim']
        self.hidden_dim = constant_params['hidden_dim']
        self.num_layer = constant_params['num_layer']
        self.bias = constant_params['bias']
        self.drop_prob = constant_params['drop_prob']

        # cnn conf
        self.conv_conf = constant_params['conv_conf']
        self.output_dim = constant_params['output_dim']

        self.spatial_lstms = []
        for i in range(self.input_size):
            self.spatial_lstms.append(
                nn.LSTM(input_size=self.input_dim,
                        hidden_size=self.hidden_dim,
                        num_layers=self.num_layer,
                        bias=self.bias,
                        batch_first=True,
                        dropout=self.drop_prob)
            )
        self.spatial_lstms = nn.ModuleList(self.spatial_lstms)

        self.output_conv = nn.Conv2d(in_channels=self.conv_conf['input_dim'],
                                     out_channels=self.output_dim,
                                     kernel_size=self.conv_conf['kernel_size'],
                                     stride=self.conv_conf['stride'],
                                     padding=self.conv_conf['kernel_size'] // 2)

        # other params
        self.regression = constant_params.get("regression", "logistic")
        self.loss_type = constant_params.get("loss_type", "BCE")
        self.clip = finetune_params['clip']
        self.lr = finetune_params['lr']
        self.set_optimizer()

        # if stateful latent info will be transferred between batches
        self.stateful = constant_params['stateful']
        self.hidden_state = None

    def reset_per_epoch(self, **kwargs):
        batch_size = kwargs['batch_size']
        self.hidden_state = self.__init_hidden(batch_size=batch_size)

    def fit(self, X, y, **kwargs):
        """
        :param X: Tensor, (b, t, m, n, d)
        :param y: Tensor, (b, t, m, n, d)
        :param kwargs:
        :return: None
        """
        X, y = Variable(X).float().to(device), Variable(y).float().to(device)
        # (b, t, m, n, d) -> (b, t, d, m, n)
        X = X.permute(0, 1, 4, 2, 3)
        y = y.permute(0, 1, 4, 2, 3)

        if self.stateful:
            # Creating new variables for the hidden state, otherwise
            # we'd back-prop through the entire training history
            self.hidden_state = list(self.__repackage_hidden(self.hidden_state))
        else:
            batch_size = X.size(0)
            self.hidden_state = self.__init_hidden(batch_size=batch_size)

        pred, self.hidden_state = self.forward(X, self.hidden_state)

        self.optimizer.zero_grad()
        loss = self.compute_loss(y, pred)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(self.parameters(), self.clip)

        # take step in classifier's optimizer
        self.optimizer.step()

        torch.cuda.empty_cache()
        return loss.detach().cpu().numpy()

    def predict(self, X):
        """
        :param X: Tensor, (b, t, m, n, d)
        :return: numpy array, b, t, m, n, d)
        """
        X = Variable(X).float().to(device)
        X = X.permute(0, 1, 4, 2, 3)

        pred, _ = self.forward(X, self.hidden_state)
        return pred.detach().cpu().numpy()

    def forward(self, input_tensor, hidden):
        """
        :param input_tensor: (B, T, D, M, N)
        :param hidden: [(n_layers, B, D), ... (n_layers, B, D)]
        :return:
        """
        b, t, d, m, n = input_tensor.shape
        flat_grid = input_tensor.view(b, t, d, m*n)

        spatial_out = []
        hidden_states = []
        for cell_idx in range(self.input_size):
            r_output, r_hidden = self.spatial_lstms(flat_grid[..., cell_idx],
                                                    hidden[cell_idx])
            hidden_states.append(r_hidden)
            spatial_out.append(r_output)
        # returning to (b, t, d, m, n)
        spatial_out = torch.stack(spatial_out, dim=3).contiguous().view(input_tensor.shape)

        output = self.output_conv(spatial_out)

        if self.regression == 'logistic':
            final_output = torch.sigmoid(output)
        else:
            final_output = output

        return final_output, hidden_states

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def compute_loss(self, labels, preds):
        """
        Computes the loss given labels and preds
        :param labels: tensor
        :param preds: tensor
        :return: loss (float)
        """
        b, t, d, m, n = preds.shape
        if self.loss_type == "MSE":
            criterion = nn.MSELoss()
            loss = criterion(preds, labels)
            return loss.mean()
        elif self.loss_type == "BCE":
            criterion = nn.BCELoss()
            preds = preds.view(b * t, d, m * n)
            labels = labels.view(b * t, d, m * n)
            return criterion(preds, labels)

    def score(self, labels, preds):
        """
        Given labels and preds arrays, compute loss
        :param labels: np array
        :param preds: np array
        :return: loss (float)
        """
        loss = self.compute_loss(torch.Tensor(labels), torch.Tensor(preds))
        return loss.numpy()

    def __init_hidden(self, batch_size):
        hidden_states = []
        for i in range(self.input_size):
            hidden = (Variable(torch.zeros(self.num_layer, batch_size, self.hidden_dim)).to(device),
                      Variable(torch.zeros(self.num_layer, batch_size, self.hidden_dim)).to(device))
            hidden_states.append(hidden)
        return hidden

    def __repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.__repackage_hidden(v) for v in h)
