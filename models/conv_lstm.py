import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = self.kernel_size // 2

        self.peephole_con = peephole_con

        if peephole_con:
            self.w_peep = Variable(torch.zeros(self.hidden_dim * 3, self.height, self.width)).to(device)

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
            batch_size = c_cur.shape[0]
            self.w_peep = self.w_peep.expand(batch_size, *self.w_peep.shape)
            w_ci, w_cf, w_co = torch.split(self.w_peep, self.hidden_dim, dim=1)
            cc_i += w_ci * c_cur
            cc_f += w_cf * c_cur

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g

        o = torch.sigmoid(cc_o) if not self.peephole_con else torch.sigmoid(cc_o + w_co * c_next)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM

        hidden = (Variable(torch.zeros(batch_size, self.hidden_dim,
                                       self.height, self.width)).to(device),
                  Variable(torch.zeros(batch_size,
                                       self.hidden_dim, self.height, self.width)).to(device))

        return hidden


class ConvLSTMBlock(nn.Module):
    def __init__(self, **kwargs):
        super(ConvLSTMBlock, self).__init__()

        # Encoder conf
        self.input_size = kwargs['input_size']
        self.input_dim = kwargs['input_dim']
        self.num_layers = kwargs['num_layers']

        # Conv-LSTM conf
        self.hidden_dim = kwargs['hidden_dim']
        self.kernel_size = kwargs['kernel_size']
        self.bias = kwargs['bias']
        self.peephole_con = kwargs['peephole_con']

        self.cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            self.cell_list += [
                ConvLSTMCell(input_size=self.input_size,
                             input_dim=cur_input_dim,
                             hidden_dim=self.hidden_dim[i],
                             kernel_size=self.kernel_size[i],
                             bias=self.bias,
                             peephole_con=self.peephole_con)
            ]
        self.cell_list = nn.ModuleList(self.cell_list)

    def init_memory(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    def forward(self, input_tensor, hidden_state):
        """
        :param input_tensor: (B, D', M, N)
        :param hidden_state: [(B, D', M, N), (B, D'', M, N), ... (B, D''', M, N)]
        :return: (B, D', M, N), hidden_states
        """
        layer_state_list = []

        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input,
                                             cur_state=hidden_state[layer_idx])
            cur_layer_input = h
            layer_state_list.append([h, c])

        return cur_layer_input, layer_state_list


class ConvLSTM(nn.Module):

    def __init__(self, constant_params, finetune_params):
        nn.Module.__init__(self)
        self.input_size = constant_params.get("input_size", (21, 41))
        self.input_dim = constant_params.get("input_dim", 5)
        self.output_dim = constant_params['output_dim']

        self.encoder_count = constant_params['encoder_count']
        self.decoder_count = constant_params['decoder_count']

        self.encoder_conf = constant_params['encoder_conf']
        self.decoder_conf = constant_params['decoder_conf']
        self.conv_conf = constant_params['conv_conf']

        # Defining blocks
        self.encoder = []
        for i in range(self.encoder_count):
            self.encoder.append(
                ConvLSTMBlock(**self.encoder_conf)
            )
        self.encoder = nn.ModuleList(self.encoder)

        self.decoder = []
        for i in range(self.decoder_count):
            self.decoder.append(
                ConvLSTMBlock(**self.decoder_conf)
            )
        self.decoder = nn.ModuleList(self.decoder)

        self.output_conv = nn.Conv2d(in_channels=self.conv_conf['input_dim'],
                                     out_channels=self.output_dim,
                                     kernel_size=self.conv_conf['kernel_size'],
                                     stride=self.conv_conf['stride'],
                                     padding=self.conv_conf['kernel_size'] // 2)

        self.regression = constant_params.get("regression", "logistic")
        self.loss_type = constant_params.get("loss_type", "BCE")

        self.detach_step = constant_params['detach_step']
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

        if kwargs['batch_idx'] % self.detach_step == 0:
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
        loss.backward(retain_graph=True)

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

    def forward(self, input_tensor, hidden_states):
        """
        :param input_tensor: (B, T, D, M, N)
        :param hidden_states: [(B, D, M, N), ..., (B, D, M, N)]
        :return: (B, T', D', M, N)
        """

        # forward encoder block
        seq_len = input_tensor.shape[1]
        cur_states = hidden_states
        for t in range(seq_len):
            # since each block corresponds to each time-step
            _, cur_states = self.encoder[t](input_tensor[:, t], cur_states)

        # reverse the state list
        cur_states = [cur_states[i-1] for i in range(len(cur_states), 0, -1)]

        # forward decoder block
        block_output_list = []
        decoder_input = torch.zeros_like(hidden_states[-1][0])
        for i in range(self.decoder_count):
            output, cur_states = self.decoder[i](decoder_input, cur_states)
            conv_output = self.output_conv(output)
            block_output_list.append(conv_output)

        block_output = torch.stack(block_output_list, dim=1)

        if self.regression == 'logistic':
            final_output = torch.sigmoid(block_output)
        else:
            final_output = block_output

        # reverse the state list
        cur_states = [cur_states[i - 1] for i in range(len(cur_states), 0, -1)]

        return final_output, cur_states

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
        # only the first block hidden is needed
        hidden_list = self.encoder[0].init_memory(batch_size)
        return hidden_list

    def __repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.__repackage_hidden(v) for v in h)
