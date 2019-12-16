import torch
import torch.nn as nn
import torch.optim as optim


from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvLSTM(nn.Module):

    class ConvLSTMCell(nn.Module):
        def __init__(self, input_size, input_dim, hidden_dim,
                     kernel_size, bias, peephole_con=False):
            """
            :param input_size: (int, int) width(M) and height(N) of input grid
            :param input_dim: int, number of channels (D) of input grid
            :param hidden_dim: int, number of channels of hidden state
            :param kernel_size: (int, int) size of the convolution kernel
            :param bias: bool weather or not to add the bias
            :param peephole_con: boolean, flag for peephole connections
            """
            super(ConvLSTM.ConvLSTMCell, self).__init__()

            self.height, self.width = input_size
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

            self.kernel_size = kernel_size
            self.bias = bias
            self.padding = self.kernel_size // 2

            self.peephole_con = peephole_con

            if peephole_con:
                self.w_peep = None

            self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

        def forward(self, input_tensor, cur_state):

            h_cur, c_cur = cur_state

            combined = torch.cat([input_tensor, h_cur], dim=1)

            combined_conv = self.conv(combined)
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

            if self.peephole_con:
                w_ci, w_cf, w_co = torch.split(self.w_peep, self.hidden_dim, dim=1)
                cc_i += w_ci * c_cur
                cc_f += w_cf * c_cur

            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            g = torch.tanh(cc_g)
            c_next = f * c_cur + i * g

            o = torch.sigmoid(cc_o + w_co * c_next) if self.peephole_con else torch.sigmoid(cc_o)
            h_next = o * torch.tanh(c_next)

            return h_next, c_next

        def init_hidden(self, batch_size):
            # Create two new tensors with sizes n_layers x batch_size x n_hidden,
            # initialized to zero, for hidden state and cell state of LSTM
            if self.peephole_con:
                self.w_peep = Variable(torch.zeros(batch_size,
                                                   self.hidden_dim * 3,
                                                   self.height, self.width)).to(device)

            hidden = (Variable(torch.zeros(batch_size, self.hidden_dim,
                                           self.height, self.width)).to(device),
                      Variable(torch.zeros(batch_size,
                                           self.hidden_dim, self.height, self.width)).to(device))

            return hidden

    def __init__(self, constant_params, finetune_params):
        nn.Module.__init__(self)

        self.finetune_params = finetune_params

        self.input_size = constant_params.get("input_size", (35, 100))
        self.input_dim = constant_params.get("input_dim", 3)

        self.regression = constant_params.get("regression", "logistic")
        self.loss_type = constant_params.get("loss_type", "BCE")

        self.height, self.width = self.input_size
        self.num_layers = constant_params['num_layers']
        self.window_len = constant_params['window_length']
        self.hidden_dim = constant_params['hidden_dim']
        self.kernel_size = constant_params['kernel_size']

        self.clip = constant_params['clip']
        self.bias = constant_params['bias']
        self.stateful = constant_params['stateful']
        self.peephole_con = constant_params['peephole_con']
        # self.connect_hidden_layers = constant_params['connect_hidden_layers']

        # Defining block
        self.cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim_en = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            self.cell_list += [self.__create_cell_unit(cur_input_dim_en, idx=i)]

        self.cell_list = nn.ModuleList(self.cell_list)

        # if stateful latent info will be transferred between batches
        self.hidden_state = None

        self.lr = finetune_params['lr']
        self.set_optimizer()

    def reset_per_epoch(self, **kwargs):
        batch_size = kwargs['batch_size']
        self.hidden_state = self.__init_hidden(batch_size=batch_size)

    def __init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

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

        pred = self.forward(X)

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

        pred = self.forward(X)
        return pred.detach().cpu().numpy()

    def score(self, labels, preds):
        """
        Given labels and preds arrays, compute loss
        :param labels: np array
        :param preds: np array
        :return: loss (float)
        """
        loss = self.compute_loss(torch.Tensor(labels), torch.Tensor(preds))
        return loss.numpy()

    def forward(self, input_tensor):
        """
        :param input_tensor: 5-D tensor of shape (b, t, m, n, d)
        :return: (b, t, m, n, d)
        """
        output_tensor, self.hidden_state = self.__forward_block(input_tensor,
                                                                self.hidden_state,
                                                                return_all_layers=True)
        output = output_tensor[-1]

        if self.regression == 'logistic':
            output = torch.sigmoid(output)

        return output

    def __forward_block(self, input_tensor, hidden_state, return_all_layers):
        """
        :param input_tensor:
        :param hidden_state:
        :param return_all_layers:
        :return: [(B, T, D, M, N), ...], [(B, D, M, N), ...] if return_all_layers false
        returns the last element of the list
        """
        layer_output_list = []
        layer_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            layer_state_list.append([h, c])

        if not return_all_layers:
            layer_output_list = layer_output_list[-1:]
            layer_state_list = layer_state_list[-1:]

        return layer_output_list, layer_state_list

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def set_finetune_params(self, new_finetune_params):
        """
        Updates finetune parameters and optimizer
        :param new_finetune_params: dict of finetune parameters
        """
        self.finetune_params = new_finetune_params
        self._set_optimizer()

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

    def __create_cell_unit(self, cur_input_dim, idx):
        cell_unit = ConvLSTM.ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[idx],
                                          kernel_size=self.kernel_size[idx],
                                          bias=self.bias,
                                          peephole_con=self.peephole_con)
        return cell_unit

    def __repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.__repackage_hidden(v) for v in h)
