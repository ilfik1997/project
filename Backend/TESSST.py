import pandas as pd
import json

with open('C:/Users/User/Desktop/ddk/processed/unique_values.json') as json_file:
    unique_values = json.load(json_file)

column_sequence = unique_values.keys()

def save_unique_train_data(
    data: pd.DataFrame, drop_columns: list, target_column: str, unique_values_path: str
) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями
    :param drop_columns: список с признаками для удаления
    :param data: датасет
    :param target_column: целевая переменная
    :param unique_values_path: путь до файла со словарем
    :return: None
    """
    unique_df = data.drop(
        columns=drop_columns + [target_column], axis=1, errors="ignore"
    )
    # создаем словарь с уникальными значениями для вывода в UI
    dict_unique = {key: unique_df[key].unique().tolist() for key in unique_df.columns}
    with open(unique_values_path, "w") as file:
        json.dump(dict_unique, file)


save_unique_train_data(data=pd.read_csv('C:/Users/User/Desktop/mac/Desktop/trneft0.csv',index_col=0),drop_columns=['Unnamed: 0'],target_column='Общее_время',unique_values_path='C:/Users/User/Desktop/ddk/processed/unique_values.json')