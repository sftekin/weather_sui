import os
import pandas as pd

from dataset import GridDataset
from torch.utils.data import DataLoader
from transformers.data_transformer import DataTransformer


def get_data(data_params):
    data = read_weather_data(data_params['data_path'],
                             data_params['data_save_path'])
    data_transformer = DataTransformer(**data_params)
    data_grid = data_transformer.transform(data)
    return data_grid


def read_weather_data(data_path, save_path):
    if os.path.isfile(data_path):
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


def create_loader(data_dict, batch_params):
    dataset_dict = {i: None for i in ['train', 'validation', 'test']}
    dataset_dict['train'] = GridDataset(data_dict['train'], cut_start=True, **batch_params)
    dataset_dict['validation'] = GridDataset(data_dict['validation'], cut_start=False, **batch_params)
    dataset_dict['test'] = GridDataset(data_dict['test'], cut_start=True, **batch_params)

    loader_dict = {i: DataLoader(dataset_dict[i],
                                 batch_size=batch_params['batch_size'],
                                 shuffle=False,
                                 num_workers=1)
                   for i in ['test', 'validation', 'train']}
    return loader_dict
