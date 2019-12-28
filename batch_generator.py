import torch


class BatchGenerator:
    def __init__(self, data, transform=None, cut_start=True, **kwargs):
        self.batch_size = kwargs['batch_size']

        # list of train and label columns, e.g [0] for temperature
        self.train_feature = kwargs['train_feature']
        self.label_feature = kwargs['label_feature']

        self.train_seq_len = kwargs['train_seq_len']
        self.label_seq_len = kwargs['label_seq_len']

        # Phase difference between train data and label
        self.phase_shift = kwargs['phase_shift']
        self.mode = kwargs['mode']

        self.transform = transform
        self.cut_start = cut_start

        # data configured  by train_len to produce label acc. to that
        self.data = self.__configure_data(data, self.train_seq_len)

        self.train_data = self.__divide_batches(self.train_seq_len, phase_shift=0)
        if self.label_seq_len < self.train_seq_len:
            self.label_data = self.__divide_batches(self.label_seq_len,
                                                    phase_shift=self.phase_shift,
                                                    step_size=self.train_seq_len)
        else:
            self.label_data = self.__divide_batches(self.label_seq_len, phase_shift=self.phase_shift)

    def __configure_data(self, data, seq_len):
        """
        :param data:(T, M, N, D)
        :return: (B, T', M, N, D)
        """
        t, m, n, d = data.shape

        # Keep only enough time steps to make full batches
        n_batches = t // (self.batch_size * seq_len)
        if self.cut_start:
            start_time_step = t - (n_batches * self.batch_size * seq_len)
            data = data[start_time_step:]
        else:
            end_time_step = n_batches * self.batch_size * seq_len
            data = data[:end_time_step]

        # Reshape into batch_size rows
        data = data.reshape((self.batch_size, -1, m, n, d))
        return data

    def __divide_batches(self, seq_len, phase_shift, step_size=None):
        if step_size is None:
            step_size = seq_len
        total_frame = self.data.shape[1]
        stacked_data = []
        for i in range(phase_shift, total_frame, step_size):
            if i+seq_len <= total_frame:
                stacked_data.append(self.data[:, i:i+seq_len])
        return stacked_data

    def __len__(self):
        if self.mode == 'train':
            return len(self.label_data)
        else:
            return len(self.train_data)

    def batch_next(self):
        """
        :return: x, y tensor in shape of (b, t, m, n, d)
        """
        if self.mode == 'train':
            for i in range(len(self.label_data)):
                x = torch.from_numpy(self.train_data[i])
                y = torch.from_numpy(self.label_data[i])
                yield x[..., self.train_feature], y[..., self.label_feature]
        else:
            for i in range(len(self.train_data)):
                x = torch.from_numpy(self.train_data[i])
                yield x[..., self.train_feature]
