from config import Params
from run_helper import get_data, dataset_split, create_loader


def run():
    params = Params()

    grid_data = get_data(params.data_params)
    data_dict = dataset_split(grid_data,
                              params.data_params['test_ratio'],
                              params.data_params['val_ratio'])

    data_loaders = create_loader(data_dict, params.train_params['batch_params'])

    for x, y in data_loaders['train']:
        a = x
        b = y

    print(a.shape)


if __name__ == '__main__':
    run()
