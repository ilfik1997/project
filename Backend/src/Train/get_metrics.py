from sklearn.metrics import mean_absolute_error, mean_squared_error

import json
import yaml

import pandas as pd
import numpy as np
from Backend.src.Train.new_metrics import r2_adjusted
from Backend.src.Train.new_metrics import mpe
from Backend.src.Train.new_metrics import mape
from Backend.src.Train.new_metrics import wape
from Backend.src.Train.new_metrics import rmsle



def create_dict_metrics(y_test: np.ndarray,
                           y_pred: np.ndarray,
                           X_test: np.ndarray)-> dict:

    """Генерация таблицы с метриками для задачи регрессии"""
    dict_metrics =  {
        'MAE' : mean_absolute_error(y_test, y_pred),
        'MSE' : mean_squared_error(y_test, y_pred),
        'RMSE' : np.sqrt(mean_squared_error(y_test, y_pred)),
        'RMSLE' : rmsle(y_test, y_pred),
        'R2 adjusted' : r2_adjusted(y_test, y_pred, X_test),
        'MPE_%' : mpe(y_test, y_pred),
        'MAPE_%' : mape(y_test, y_pred),
        'WAPE_%' : wape(y_test, y_pred),
    }
    return dict_metrics




def save_metrics(
    X_test: pd.DataFrame, y_test: pd.Series, model: object, metric_path: str
) -> None:
    """
    Получение и сохранение метрик
    :param data_x: объект-признаки
    :param data_y: целевая переменная
    :param model: модель
    :param metric_path: путь для сохранения метрик
    """
    result_metrics = create_dict_metrics(
        y_test=y_test,
        y_pred=model.predict(X_test),
        X_test=X_test
    )
    with open(metric_path, "w") as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    Получение метрик из файла
    :param config_path: путь до конфигурационного файла
    :return: метрики
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config["train"]["metrics_path"]) as json_file:
        metrics = json.load(json_file)

    return metrics