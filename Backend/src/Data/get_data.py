import pandas as pd
from typing import Text


def get_dataset(dataset_path: Text)-> pd.DataFrame:
    '''
    чтение датасета из csv формата
    :param dataset_path: путь к таблице csv
    :return:
    '''
    return pd.read_csv(dataset_path, index_col=0)

