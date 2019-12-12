import os
import pandas as pd

from config import Params
from transformers.data_transformer import DataTransformer


def run():
    params = Params()
    data_transf = DataTransformer(**params.data_params)
    data = data_transf.get_data()
    print()

if __name__ == '__main__':
    run()
