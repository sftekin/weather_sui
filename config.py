
import pandas as pd


class Params:
    def __init__(self):

        self.data_params = {
            'start_date': pd.to_datetime('01-01-2017 00:00:00', format='%d-%m-%Y %H:%M:%S'),
            'end_date': pd.to_datetime('27-03-2018 00:00:00', format='%d-%m-%Y %H:%M:%S'),
            'freq': 3,
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
                    'batch_size': 1,
                    'train_feature': [0, 1, 2, 3, 4],
                    'label_feature': [0],
                    'train_seq_len': 30,
                    'label_seq_len': 5,
                    'phase_shift': 1,  # Phase difference between train data and label
                    'mode': 'train',
                },
                'constant_params': {
                    'window_in': 10,  # Same with train_seq_len
                    'window_out': 5,  # Same with label_seq_len
                    'detach_step': 1,
                    'output_dim': 1,
                    'encoder_conf': {
                        'input_size': (21, 41),
                        'input_dim': 5,
                        'num_layers': 5,
                        'hidden_dim': [5, 16, 16, 32, 64],
                        'kernel_size': [5, 5, 5, 5, 5],
                        'peephole_con': False,
                        'bias': True,
                        'return_all_layers': False
                    },
                    'decoder_conf': {
                        'input_size': (21, 41),
                        'input_dim': 64,
                        'num_layers': 5,
                        'hidden_dim': [64, 32, 16, 16, 5],
                        'kernel_size': [3, 3, 3, 3, 3],
                        'peephole_con': False,
                        'bias': True,
                        'return_all_layers': False
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
            'SpatialLSTM': {
                'batch_params': {
                    'batch_size': 5,
                    'train_feature': [0, 1, 2, 3, 4],
                    'label_feature': [0],
                    'train_seq_len': 10,
                    'label_seq_len': 10,
                    'phase_shift': 1,  # Phase difference between train data and label
                    'mode': 'train',
                },
                'constant_params': {
                    'input_size': (21, 41),
                    'seq_len': 10,
                    'input_dim': 5,
                    'hidden_dim': 5,
                    'num_layer': 2,
                    'bias': True,
                    'drop_prob': 0.3,
                    'output_dim': 1,
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
            'EMA': {
                'batch_params': {
                    'batch_size': 1,
                    'train_feature': [0],
                    'label_feature': [0],
                    'train_seq_len': 5,
                    'label_seq_len': 5,
                    'phase_shift': 5,  # Phase difference between train data and label
                    'mode': 'train',
                },
                'constant_params': {
                    'window_len': 5,  # Must be same with label_seq_len
                    'label_feature': [0]
                },
                'finetune_params': {
                    'mu': .1,
                    'epoch': 1
                }
            },
            'SMA': {
                'batch_params': {
                    'batch_size': 1,
                    'train_feature': [0],
                    'label_feature': [0],
                    'train_seq_len': 5,
                    'label_seq_len': 5,
                    'phase_shift': 5,  # Phase difference between train data and label
                    'mode': 'train',
                },
                'constant_params': {
                    'window_len': 5,
                    'output_len': 5,  # Must be same with label_seq_len
                    'train_weights': True,
                    'attention_to': 'middle',
                    'init_dist': 'uniform'
                },
                'finetune_params': {
                    "lr": 0.01,
                    'epoch': 200,

                }
            }
        }
