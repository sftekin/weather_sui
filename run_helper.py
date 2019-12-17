import os
import pandas as pd

from batch_generator import BatchGenerator
from transformers.data_transformer import DataTransformer


def get_data(data_params):
    data = _read_weather_data(data_params['data_path'],
                              data_params['data_save_path'])
    data_transformer = DataTransformer(**data_params)
    data_grid = data_transformer.transform(data)
    return data_grid


def _read_weather_data(data_path, save_path):
    if os.path.isfile(save_path):
        print('Loading data from pickle')
        data = pd.read_pickle(save_path)
    else:
        print('Reading CSV File...')
        data = pd.read_csv(data_path)
        data.to_pickle(save_path)
    return data


def dataset_split(grid_data, test_ratio=0.1, val_ratio=0.1):
    n_data = grid_data.shape[0]

    test_index = n_data - int(n_data * test_ratio)
    val_index = test_index - int(n_data * val_ratio)

    data_dict = dict()
    data_dict['test'] = grid_data[test_index:]
    data_dict['validation'] = grid_data[val_index:test_index]
    data_dict['train'] = grid_data[:val_index]
    return data_dict


def create_generator(data_dict, batch_params):
    batch_gen_dict = {i: None for i in ['train', 'validation', 'test']}
    batch_gen_dict['validation'] = BatchGenerator(data_dict['validation'], cut_start=False, **batch_params)
    batch_gen_dict['train'] = BatchGenerator(data_dict['train'], cut_start=True, **batch_params)
    batch_gen_dict['test'] = BatchGenerator(data_dict['test'], cut_start=True, **batch_params)
    return batch_gen_dict
