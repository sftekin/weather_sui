
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
                    'batch_size': 1,
                    'sequence_len': 96,
                    'output_feature': [0],
                    'output_frame': list(range(96)),
                    'input_feature': [0, 1, 2, 3, 4],
                    'step_size': 96,  # step difference between batches
                    'mode': 'train',
                    'shift_size': 1  # distance btw y and x, only train mode
                },
                'constant_params': {
                    'input_size': (21, 41),
                    'input_dim': 5,
                    'num_layers': 5,
                    'window_length': 32,  # This should be same with batch config
                    'hidden_dim': [5, 20, 20, 10, 1],
                    'kernel_size': [5, 5, 5, 3, 3],
                    'clip': 5,
                    'bias': True,
                    'stateful': True,
                    'peephole_con': False,
                    "regression": "regression",
                    "loss_type": "MSE"
                },
                'finetune_params': {
                    "lr": 0.00001,
                    'epoch': 50,
                }
            },
            'EMA': {
                'batch_params': {
                    'batch_size': 1,
                    'sequence_len': 240,
                    'output_feature': [0],
                    'input_feature': [0],
                    'output_frame': [-1],
                    'step_size': 240,  # step difference between batches
                    'mode': 'train',
                    'shift_size': 1  # distance btw y and x, only train mode
                },
                'constant_params': {
                    'window_len': 240
                },
                'finetune_params': {
                    'mu': .1,
                    'epoch': 50
                }
            },
            'SMA': {
                'batch_params': {
                    'batch_size': 1,
                    'sequence_len': 240,
                    'output_feature': [0],
                    'input_feature': [0],
                    'output_frame': [-1],
                    'step_size': 240,  # step difference between batches
                    'mode': 'train',
                    'shift_size': 1  # distance btw y and x, only train mode
                },
                'constant_params': {
                    'window_len': 240,
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
