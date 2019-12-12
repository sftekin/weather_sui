
import pandas as pd


class Params:
    def __init__(self):

        self.data_params = {
            'start_date': pd.to_datetime('01-01-2018', format='%d-%m-%Y'),
            'end_date': pd.to_datetime('01-03-2018', format='%d-%m-%Y'),
            'freq': 3,
            'normalise_events': False
        }
