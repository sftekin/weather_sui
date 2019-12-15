import pickle

from config import Params
from run_helper import get_data, dataset_split, create_loader
from train import trainer


def run():
    params = Params()

    grid_data = get_data(params.data_params)
    data_dict = dataset_split(grid_data,
                              params.data_params['test_ratio'],
                              params.data_params['val_ratio'])

    data_loaders = create_loader(data_dict, params.model_params['batch_params'])
    trained_model = trainer(data_loaders, **params.model_params)

    print('Training finished, saving the model')
    model_file = open('results/conv_lstm.pkl', 'wb')
    pickle.dump(trained_model, model_file)


if __name__ == '__main__':
    run()
