import numpy as np
import torch

from torch.utils.data import Dataset


class GridDataset(Dataset):
    def __init__(self, grid, transform=None, cut_start=True, **kwargs):
        self.seq_len = kwargs['sequence_len']
        self.batch_size = kwargs['batch_size']
        self.transform = transform
        self.cut_start = cut_start
        self.n_time_step = grid.shape[0]
        self.grid = self.configure_data(grid)

    def configure_data(self, data):
        """
        :param data:(T, M, N, D)
        :return: (B, T', M, N, D)
        """
        t, m, n, d = data.shape

        # Keep only enough time steps to make full batches
        n_batches = self.n_time_step // (self.batch_size * self.seq_len)
        if self.cut_start:
            start_time_step = self.n_time_step - (n_batches * self.batch_size * self.seq_len)
            data = data[start_time_step:]
        else:
            end_time_step = n_batches * self.batch_size * self.seq_len
            data = data[:end_time_step]

        # Reshape into batch_size rows
        data = data.reshape((self.batch_size, -1, m, n, d))
        return data

    def __len__(self):
        return self.grid.shape[1]

    def __getitem__(self, idx):
        batch_idx = idx % self.seq_len
        seq_idx = idx - batch_idx
        selected_grid = self.grid[batch_idx, seq_idx:seq_idx+self.seq_len]

        label = np.zeros_like(selected_grid)
        try:
            label[:-1], label[-1] = selected_grid[1:], self.grid[batch_idx, seq_idx+self.seq_len]
        except IndexError:
            label[:-1], label[-1] = selected_grid[1:], self.grid[batch_idx, 0]

        selected_grid = torch.from_numpy(selected_grid)
        label = torch.from_numpy(label[:, :, :, :1])

        return selected_grid, label
