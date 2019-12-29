
import pandas as pd


class Params:
    def __init__(self):

        self.data_params = {
            'start_date': pd.to_datetime('01-01-2017 00:00:00', format='%d-%m-%Y %H:%M:%S'),
            'end_date': pd.to_datetime('27-03-2018 00:00:00', format='%d-%m-%Y %H:%M:%S'),
            'freq': 1,
            'normalise_events': False,
            'data_path': 'data/London_historical_meo_grid.csv',
            'data_save_path': 'data/pickles/raw_data.pkl',
            'grid_save_path': 'data/pickles/grid_data',
            'test_ratio': 0.1,
            'val_ratio': 0.1,
        }

        self.run_params = {
            'model_list': ['CONVLSTM', 'EMA', 'SMA']
        }

        self.model_params = {
            'CONVLSTM': {
                'batch_params': {
                    'batch_size': 4,
                    'train_feature': [0, 1, 2, 3, 4],
                    'label_feature': [0],
                    'train_seq_len': 10,
                    'label_seq_len': 15,
                    'phase_shift': 1,  # Phase difference between train data and label
                    'mode': 'train',
                },
                'constant_params': {
                    'encoder_count': 10,  # Same with train_seq_len
                    'decoder_count': 15,  # Same with label_seq_len
                    'detach_step': 5,
                    'output_dim': 1,
                    'encoder_conf': {
                        'input_size': (21, 41),
                        'input_dim': 5,
                        'num_layers': 5,
                        'hidden_dim': [5, 16, 16, 32, 64],
                        'kernel_size': [5, 5, 5, 5, 5],
                        'peephole_con': False,
                        'bias': True,
                    },
                    'decoder_conf': {
                        'input_size': (21, 41),
                        'input_dim': 64,
                        'num_layers': 5,
                        'hidden_dim': [64, 32, 16, 16, 5],
                        'kernel_size': [3, 3, 3, 3, 3],
                        'peephole_con': False,
                        'bias': True,
                    },
                    'conv_conf': {
                        'input_dim': 5,
                        'kernel_size': 1,
                        'stride': 1,
                    },
                    'stateful': True,
                    "regression": "regression",
                    "loss_type": "MSE"
                },
                'finetune_params': {
                    'clip': 5,
                    "lr": 0.001,
                    'epoch': 50,
                }
            },
            'TRAJGRU': {
                'batch_params': {
                    'batch_size': 5,
                    'train_feature': [0, 1, 2, 3, 4],
                    'label_feature': [0],
                    'train_seq_len': 5,
                    'label_seq_len': 1,
                    'phase_shift': 1,  # Phase difference between train data and label
                    'mode': 'train',
                },
                'constant_params': {
                    'encoder_count': 5,
                    'decoder_count': 1,
                    'stateful': True,
                    'encoder_conf': {
                        'input_size': (21, 41),
                        'input_dim': 5,
                        'num_layers': 2,
                        'conv_dims': [16, 64],
                        'conv_kernel': 3,
                        'conv_stride': 1,
                        'pool_kernel': 3,
                        'pool_stride': 2,
                        'pool_padding': 0,
                        'gru_dims': [32, 96],
                        'gru_kernels': [5, 3],
                        'connection': 5,
                        'bias': True
                    },
                    'decoder_conf': {
                        'input_size': (6, 11),
                        'input_dim': 96,
                        'output_dim': 1,
                        'num_layers': 2,
                        'conv_dims': [64, 16],
                        'conv_kernel': 3,
                        'conv_stride': 2,
                        'conv_padding': 0,
                        'gru_dims': [96, 32],
                        'gru_kernels': [3, 3],
                        'connection': 5,
                        'bias': True
                    }
                },
                'finetune_params': {
                    "lr": 0.0001,
                    'epoch': 200,
                    'clip': 5
                }
            },
            'EMA': {
                'batch_params': {
                    'batch_size': 5,
                    'train_feature': [0, 1, 2, 3, 4],
                    'label_feature': [0],
                    'train_seq_len': 15,
                    'label_seq_len': 15,
                    'phase_shift': 1,  # Phase difference between train data and label
                    'mode': 'train',
                },
                'constant_params': {
                    'window_len': 5
                },
                'finetune_params': {
                    'mu': .1,
                    'epoch': 50
                }
            },
            'SMA': {
                'batch_params': {
                    'batch_size': 5,
                    'train_feature': [0],
                    'label_feature': [0],
                    'train_seq_len': 5,
                    'label_seq_len': 1,
                    'phase_shift': 1,  # Phase difference between train data and label
                    'mode': 'train',
                },
                'constant_params': {
                    'window_len': 5,
                    'train_weights': True,
                    'attention_to': 'right',
                    'init_dist': 'kaiser'
                },
                'finetune_params': {
                    "lr": 0.001,
                    'epoch': 30,

                }
            }
        }
