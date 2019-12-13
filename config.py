
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
