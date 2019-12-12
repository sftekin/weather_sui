import os
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


class DataTransformer:
    def __init__(self, **kwargs):

        self.start_date = kwargs['start_date']
        self.end_date = kwargs['end_date']
        self.freq = kwargs['freq']

        self.data_path = 'data/London_historical_meo_grid.csv'
        self.save_path = 'data/pickles/raw_data.pkl'

    def get_data(self):
        data_df = self.__read_weather_data()
        data_df = self.__transform_weather_data(data_df)
        return data_df

    def __read_weather_data(self):
        if os.path.isfile(self.save_path):
            print('Loading from pickle')
            data = pd.read_pickle(self.save_path)
        else:
            print('Reading CSV File...')
            data = pd.read_csv(self.data_path)
            data.to_pickle(self.save_path)
        return data

    def __transform_weather_data(self, data):
        data = data.rename(index=str, columns={'stationName': 'grid_index',
                                               'utc_time': 'date'})
        # from string to int for grid ids
        def to_grid(x):
            num = int(x.split('_')[-1])
            return num
        data['grid_index'] = data.loc[:, 'grid_index'].map(to_grid)

        # crop the date for input data
        data_cropped = self.__crop_dates(data)

        # degrade by input freq
        grid_indexes = np.unique(data_cropped['grid_index'])
        data_avg = self.__freq_mean(data_cropped, grid_indexes)
        return data_avg

    def __crop_dates(self, data):
        date_range = pd.to_datetime(data['date'])
        if self.start_date < date_range.iloc[0]:
            raise Exception('input start_date not in the date range')
        elif self.end_date > date_range.iloc[-1]:
            raise Exception('input end_date not in the date range')
        else:
            data_crop = data.loc[(self.end_date >= date_range) & (date_range >= self.start_date)]

        data_crop.loc[:, 'date'] = pd.to_datetime(data_crop['date'])
        data_crop = data_crop.reset_index(drop=True)
        return data_crop

    def __freq_mean(self, data, grid_indexes):
        data_avg = pd.DataFrame()
        for idx in grid_indexes:
            data_stat = data.loc[data['grid_index'] == idx]
            data_stat = data_stat.groupby(pd.Grouper(key='date', freq=str(self.freq) + 'H')).mean()
            data_avg = data_avg.append(data_stat)
        data_avg.reset_index(inplace=True)
        return data_avg
