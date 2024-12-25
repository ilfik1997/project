"""
Программа: Frontend часть проекта
Версия: 1.0
"""
import os

import yaml
import streamlit as st
from src.data.get_data import load_data
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_input, evaluate_from_file



CONFIG_PATH = "C:/Users/User/Desktop/ddk/config/params.yml"


def main_page():
    """
    Страница с описанием проекта
    """
    st.image(
        'https://vectorseek.com/wp-content/uploads/2024/02/Transneft-rus-Logo-Vector.svg-.png',
        width=600,
    )

    st.markdown("# Описание проекта")
    st.title("MLOps project: Предсказание длительности бизнес-процессов в компании Транснефть")
    st.write(
        """
        Имеется база данных по предыдущим работам(бизнес-процессам), проведенных за 2023-2024 года, 
        необходимо предсказать длительность следующих работ для последующего анализа результативности 
        той или иной бригады."""
    )

    # name of the columns
    st.markdown(
        """
        ### Описание полей 
            - Общее время:
            - Количество персонала:
            - Количество оборудования:
            - Количество заявок у диспетчера:
            - Количество диспетчеров:
            - Количество свободных машин:
            - Количество персонала УБ:
            - Средняя скорость машин:
            - Время остановок:
            - Расстояние:
            - опыт мастера ЛАЭС, лет:
            - Количество выездов на участок:
            - Опыт экскаваторщика, лет:
            - Глубина вкрытия котлована, м:
            - Количество персонала для вскрытия:
            - Количество персонала.1:
            - Параметры дефекта (длина):
            - Параметры дефекта (ширина):
            - Количество работников:
            - Длина для нанесения изоляции:
            - Количество персонала: 
            - Объем засыпки, м3:
            - Время года:
            - Тип изоляции:
            - Обводненность:
            - Классификация грунта:
            - Ручной труд/тип экскаватора:      
    """
    )


"def exploratory(): Заполнить !!!!!"





def training():
    """
    Тренировка модели
    """
    st.markdown("# Training model CatBoost")
    # get params
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]

    if st.button("Обучение модели"):
        start_training(config=config, endpoint=endpoint)


def prediction():
    """
    Получение предсказаний путем ввода данных
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_input"]
    unique_data_path = config["preprocessing"]["unique_values_path"]

    # проверка на наличие сохраненной модели
    if os.path.exists(config["train"]["model_path"]):
        evaluate_input(unique_data_path=unique_data_path, endpoint=endpoint)
    else:
        st.error("Сначала обучите модель")


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]

    upload_file = st.file_uploader(
        "", type=["csv", "xlsx"], accept_multiple_files=False
    )
    # проверка загружен ли файл
    if upload_file:
        dataset_csv_df, files = load_data(data=upload_file, type_data="Test")
        # проверка на наличие сохраненной модели
        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
        else:
            st.error("Сначала обучите модель")




def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Тренировка модели": training,
        "Предсказание": prediction,
        "Предсказание из файла": prediction_from_file,
    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()



if __name__ == "__main__":
    main()
