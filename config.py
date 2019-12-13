
import pandas as pd


class Params:
    def __init__(self):

        self.data_params = {
            'start_date': pd.to_datetime('01-01-2018 00:00:00', format='%d-%m-%Y %H:%M:%S'),
            'end_date': pd.to_datetime('01-03-2018 00:00:00', format='%d-%m-%Y %H:%M:%S'),
            'freq': 3,
            'normalise_events': False,
            'data_path': 'data/London_historical_meo_grid.csv',
            'data_save_path': 'data/pickles/raw_data.pkl',
            'grid_save_path': 'data/pickles/grid_data',
            'test_ratio': 0.1,
            'val_ratio': 0.1,
        }

        self.train_params = {
            'batch_params': {
                'batch_size': 4,
                'sequence_len': 4,
            },
            'trainer_params': {

            }
        }

        self.model_params = {
            'constant_params': {
                "input_size": (35, 100),
                "input_dim": 13,
                "output_dim": 6,
                'num_layers': 5,
                'window_length': 8,  # This should be same with batch config
                'encoder_hidden_dim': [20, 20, 10],
                'encoder_kernel_size': [5, 5, 5, 3, 3],
                'decoder_hidden_dim': [10, 10, 10],
                'decoder_kernel_size': [5, 5, 5, 3, 3],
                'clip': 5,
                'bias': True,
                'stateful': True,
                'peephole_con': False,
                "regression": "regression",
                "loss_type": "MSE"
            },
            'finetune_params': {
                "lr": 0.001,
            }
        }
