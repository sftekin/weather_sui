import numpy as np
import torch


class BatchGenerator:
    def __init__(self, grid, transform=None, cut_start=True, **kwargs):
        self.output_feature = kwargs['output_feature']  # list of label columns, e.g [0] for temperature
        self.input_feature = kwargs['input_feature']  # list of label columns, e.g [0] for temperature
        self.output_frame = kwargs['output_frame']
        self.batch_size = kwargs['batch_size']
        self.seq_len = kwargs['sequence_len']
        self.step_size = kwargs['step_size']
        self.mode = kwargs['mode']
        self.shift_size = kwargs['shift_size']

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
        data = data[:, :, :, :, self.input_feature]
        return data

    def __len__(self):
        return self.n_time_step

    def batch_next(self):
        """
        :return: x, y tensor in shape of (b, t, m, n, d)
        """
        if self.mode == 'train':
            yield from self._train_loop()
        else:
            yield from self._pred_loop()

    def _train_loop(self):
        for n in range(0, self.n_time_step, self.step_size):
            y_idx = n + self.shift_size
            if y_idx+self.seq_len < self.n_time_step:
                x = self.grid[:, n:n+self.seq_len]
                y = self.grid[:, y_idx:y_idx+self.seq_len]

                x = torch.from_numpy(x)
                y = torch.from_numpy(y)
                y = y[:, :, :, :, self.output_feature]
                y = y[:, self.output_frame]
                yield x, y

    def _pred_loop(self):
        for n in range(0, self.n_time_step, self.step_size):
            if n + self.seq_len < self.n_time_step:
                x = self.grid[:, n:n+self.seq_len]
                yield x

