import os
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


class DataTransformer:
    def __init__(self, **kwargs):
        self.start_date = kwargs['start_date']
        self.end_date = kwargs['end_date']
        self.freq = kwargs['freq']
        self.T = pd.date_range(start=self.start_date,
                               end=self.end_date,
                               freq=str(self.freq) + 'H')
        self.M = None
        self.N = None
        self.D = None

        start_date_str = self.start_date.strftime('%d-%m-%Y')
        end_date_str = self.end_date.strftime('%d-%m-%Y')
        self.grid_save_path = kwargs['grid_save_path'] + (start_date_str +
                                                          '_' + end_date_str +
                                                          '_' + str(self.freq) +
                                                          '.npy')

    def transform(self, data_df):
        if os.path.isfile(self.grid_save_path):
            print('Grid has found loading...')
            data_grid = np.load(self.grid_save_path)
        else:
            # Define the rectangle
            data_columns = list(data_df.columns.values)
            selected_columns = data_columns[data_columns.index('temperature'):]
            self.D = len(selected_columns)
            self.M = len(np.unique(data_df['latitude']))
            self.N = len(np.unique(data_df['longitude']))

            # Crop by date and degrade by freq
            data_df = self.__transform_weather_data(data_df)

            # Convert to numpy array in T, M, N, D shape
            data_grid = self.__transform_grid(data_df)

            print('Grid created saving..')
            np.save(self.grid_save_path, data_grid)
        return data_grid

    def __transform_weather_data(self, data):
        data = data.rename(index=str, columns={'stationName': 'grid_index',
                                               'utc_time': 'date'})

        # from string to int for grid ids
        def to_grid_idx(x):
            num = int(x.split('_')[-1])
            return num
        data['grid_index'] = data.loc[:, 'grid_index'].map(to_grid_idx)

        # crop the date for input data
        data_cropped = self.__crop_dates(data)

        # degrade by input freq
        if self.freq > 1:
            grid_indexes = np.unique(data_cropped['grid_index'])
            data_cropped = self.__freq_mean(data_cropped, grid_indexes)
        return data_cropped

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

    def __transform_grid(self, data_df):
        grid_array = np.zeros((len(self.T), self.M, self.N, self.D))
        for day in range(len(self.T)):
            selected_data = data_df[data_df['date'] == self.T[day]]
            selected_data = selected_data.loc[:, 'temperature':].values
            selected_data = np.flip(selected_data.reshape((self.M, self.N, self.D), order='F'), axis=0)
            grid_array[day, :] = selected_data
        return grid_array




