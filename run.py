import os
import pandas as pd

from config import Params
from transformers.data_transformer import DataTransformer


def read_weather_data(data_path, save_path):
    if os.path.isfile(data_path):
        print('Loading from pickle')
        data = pd.read_pickle(save_path)
    else:
        print('Reading CSV File...')
        data = pd.read_csv(data_path)
        data.to_pickle(save_path)
    return data


def run():
    params = Params()
    data_transf = DataTransformer(**params.data_params)
    data = read_weather_data(params.data_params['data_path'],
                             params.data_params['save_path'])
    grid_data = data_transf.transform(data)
    print()


if __name__ == '__main__':
    run()
