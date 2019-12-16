import numpy as np
import torch


class BatchGenerator:
    def __init__(self, grid, transform=None, cut_start=True, **kwargs):
        self.output_feature = kwargs['output_feature']
        self.seq_len = kwargs['sequence_len']
        self.batch_size = kwargs['batch_size']
        self.transform = transform
        self.cut_start = cut_start
        self.grid = self.configure_data(grid)
        self.n_time_step = self.grid.shape[1]

    def configure_data(self, data):
        """
        :param data:(T, M, N, D)
        :return: (B, T', M, N, D)
        """
        t, m, n, d = data.shape

        # Keep only enough time steps to make full batches
        n_batches = t // (self.batch_size * self.seq_len)
        if self.cut_start:
            start_time_step = t - (n_batches * self.batch_size * self.seq_len)
            data = data[start_time_step:]
        else:
            end_time_step = n_batches * self.batch_size * self.seq_len
            data = data[:end_time_step]

        # Reshape into batch_size rows
        data = data.reshape((self.batch_size, -1, m, n, d))
        return data

    def __len__(self):
        return self.n_time_step

    def batch_next(self):
        """
        :param feature_idx: list of label columns, e.g [0] for temperature
        :return: x, y tensor in shape of (b, t, m, n, d)
        """
        # TODO: Add step-size I dont want it to be seq_len always
        # TODO: shift y value n times and prediction and train mode should be specified
        for n in range(0, self.n_time_step, self.seq_len):
            x = self.grid[:, n:n+self.seq_len]

            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], self.grid[:, n+self.seq_len]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], self.grid[:, 0]

            x = torch.from_numpy(x)
            y = torch.from_numpy(y[:, :, :, :, self.output_feature])

            yield x, y
