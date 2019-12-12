import os
import re
import numpy as np
import pandas as pd


class DataTransformer:
    def __init__(self, **kwargs):

        self.start_date = kwargs['start_date']
        self.end_date = kwargs['end_date']
        self.freq = kwargs['freq']

        self.data_path = '/data/London_historical_meo_grid'

    def get_data(self):
        data = self.__read_weather_data()
        return data

    def __read_weather_data(self):
        if os.path.isfile(self.data_path + '.pkl'):
            print('Loading from pickle')
            temp_data = pd.read_pickle(self.data_path + '.pkl')
        else:
            print('Reading Excel Files...')
            temp_data = pd.read_csv(self.data_path + '.csv')

            temp_data.to_pickle(self.data_path + '.pkl')
        return temp_data
