import os
import yaml
import joblib
import pandas as pd
from Backend.src.Data.get_data import  get_dataset
from Backend.src.Transform.Transform import pipeline_preprocess


def pipeline_evaluate(
    config_path, dataset: pd.DataFrame = None, data_path: str = None
) -> list:
    """
    Предобработка входных данных и получение предсказаний
    :param dataset: датасет
    :param config_path: путь до конфигурационного файла
    :param data_path: путь до файла с данными
    :return: предсказания
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preprocessing_config = config["preprocessing"]
    train_config = config["train"]

    # preprocessing
    if data_path:
        dataset = get_dataset(dataset_path=data_path)

    dataset = pipeline_preprocess(data=dataset, **preprocessing_config)

    model = joblib.load(os.path.join(train_config["model_path"]))
    prediction = model.predict(dataset).tolist()

    return prediction


