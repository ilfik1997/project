from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test(dataset: pd.DataFrame, **kwargs):
    """
    Разделение данных на train/test с последующим сохранением
    :param dataset: датасет
    :return: train/test датасеты
    """
    # Split in train/test
    df_train, df_test = train_test_split(

        dataset,
        test_size=kwargs["test_size"],
        random_state=kwargs["random_state"],
    )
    df_train.to_csv(kwargs["train_path_proc"], index=False)
    df_test.to_csv(kwargs["test_path_proc"], index=False)
    return df_train, df_test

def get_train_test_data(
    data_train: pd.DataFrame, data_test: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Получение train/test данных разбитых по отдельности на объект-признаки и целевую переменную
    :param data_train: train датасет
    :param data_test: test датасет
    :param target: название целевой переменной
    :return: набор данных train/test
    """
    X_train, X_test = (
        data_train.drop(target, axis=1),
        data_test.drop(target, axis=1),
    )
    y_train, y_test = (
        data_train.loc[:, target],
        data_test.loc[:, target],
    )
    return X_train, X_test, y_train, y_test


def split_val(X_train,y_train):
    '''
    разделение тестовых данных на тестовые_1 и валидационные
    :param X_train:
    :param y_train:
    :return:
    '''
    X_train_, X_val, y_train_, y_val = train_test_split(X_train,
                                                        y_train,
                                                        test_size=0.16,
                                                        shuffle=True,
                                                        random_state=10)
    return X_train_, X_val, y_train_, y_val
