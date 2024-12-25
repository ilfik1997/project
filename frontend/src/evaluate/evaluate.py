"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 1.0
"""

import json
from io import BytesIO
import pandas as pd
import requests
import streamlit as st
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def evaluate_input(unique_data_path: str, endpoint: object) -> None:
    """
    Получение входных данных путем ввода в UI -> вывод результата
    :param unique_data_path: путь до уникальных значений
    :param endpoint: endpoint
    """
    with open(unique_data_path) as file:
        unique_df = json.load(file)

    # поля для вводы данных, используем уникальные значения
    Общее_время = st.sidebar.number_input("Общее_время",
                                                   min_value=0,
                                                   max_value=1000
                                                   )

    Количество_персонала = st.sidebar.number_input("Количество_персонала",
        min_value=min(unique_df["Количество_персонала"]),
        max_value=max(unique_df["Количество_персонала"]),
    )
    Количество_оборудования = st.sidebar.number_input("Количество_оборудования",
        min_value=min(unique_df["Количество_оборудования"]),
        max_value=max(unique_df["Количество_оборудования"]),
    )
    Количество_заявок_у_диспетчера = st.sidebar.number_input("Количество_заявок_у_диспетчера",
        min_value=min(unique_df["Количество_заявок_у_диспетчера"]),
        max_value=max(unique_df["Количество_заявок_у_диспетчера"]),
    )
    Количество_диспетчеров = st.sidebar.selectbox("Количество_диспетчеров",
        (unique_df["Количество_диспетчеров"])
    )
    Количеcтво_свободных_машин = st.sidebar.selectbox("Количество_свободных_машин",
        (unique_df["Количество_свободных_машин"])
    )
    Количество_персонала_УБ = st.sidebar.selectbox("Количество_персонала_УБ",
        (unique_df["Количество_персонала_УБ"])
    )
    Средняя_скорость_машин=st.sidebar.slider("Средняя_скорость_машин",
        min_value=min(unique_df["Средняя_скорость_машин"]),
        max_value=max(unique_df["Средняя_скорость_машин"]),
    )
    Время_остановок=st.sidebar.selectbox("Время_остановок",
        (unique_df["Время_остановок"])
    )
    Расстояние = st.sidebar.number_input("Расстояние",
        min_value=min(unique_df["Расстояние"]),
        max_value=max(unique_df["Расстояние"]),
    )
    Опыт_мастера_ЛАЭС_лет = st.sidebar.number_input("Опыт_мастера_ЛАЭС_лет",
        min_value=min(unique_df["Опыт_мастера_ЛАЭС_лет"]),
        max_value=max(unique_df["Опыт_мастера_ЛАЭС_лет"]),
    )
    Количество_выездов_на_участок =st.sidebar.number_input("Количество_выездов_на_участок",
        min_value=min(unique_df["Количество_выездов_на_участок"]),
        max_value=max(unique_df["Количество_выездов_на_участок"]),
    )
    Опыт_экскаваторщика_лет =st.sidebar.number_input("Опыт_экскаваторщика_лет",
        min_value=min(unique_df["Опыт_экскаваторщика_лет"]),
        max_value=max(unique_df["Опыт_экскаваторщика_лет"]),
    )
    Ручной_труд_тип_экскаватора= st.sidebar.selectbox("Ручной_труд_тип_экскаватора",
        (unique_df["Ручной_труд_тип_экскаватора"])
    )
    Глубина_вcкрытия_котлована_м = st.sidebar.slider("Глубина_вскрытия_котлована_м",
        min_value=min(unique_df["Глубина_вскрытия_котлована_м"]),
        max_value=max(unique_df["Глубина_вскрытия_котлована_м"]),
    )
    Классификация_грунта= st.sidebar.selectbox("Классификация_грунта",
        (unique_df["Классификация_грунта"])
    )
    Обводненность= st.sidebar.selectbox("Обводненность",
        (str(0),str(1))
    )
    Количество_персонала_для_вскрытия= st.sidebar.slider("Количество_персонала_для_вскрытия",
        min_value=min(unique_df["Количество_персонала_для_вскрытия"]),
        max_value=max(unique_df["Количество_персонала_для_вскрытия"]),
    )
    Тип_изоляции= st.sidebar.selectbox("Тип_изоляции",
        (unique_df["Тип_изоляции"])
    )
    Количество_персонала1= st.sidebar.number_input("Количество_персонала1",
        min_value=min(unique_df["Количество_персонала1"]),
        max_value=max(unique_df["Количество_персонала1"]),
    )
    Параметры_дефекта_длина = st.sidebar.number_input("Параметры_дефекта_длина",
        min_value=min(unique_df["Параметры_дефекта_длина"]),
        max_value=max(unique_df["Параметры_дефекта_длина"]),
    )
    Параметры_дефекта_ширина = st.sidebar.number_input("Параметры_дефекта_ширина",
        min_value=min(unique_df["Параметры_дефекта_ширина"]),
        max_value=max(unique_df["Параметры_дефекта_ширина"]),
    )
    Количество_работников= st.sidebar.number_input("Количество_работников",
        min_value=min(unique_df["Количество_работников"]),
        max_value=max(unique_df["Количество_работников"]),
    )
    Длина_для_нанесения_изоляции = st.sidebar.number_input("Длина_для_нанесения_изоляции",
        min_value=min(unique_df["Длина_для_нанесения_изоляции"]),
        max_value=max(unique_df["Длина_для_нанесения_изоляции"]),
    )
    Количество_персонала_= st.sidebar.number_input("Количество_персонала_",
        min_value=min(unique_df["Количество_персонала_"]),
        max_value=max(unique_df["Количество_персонала_"]),
    )
    Объем_засыпки_м3= st.sidebar.number_input("Объем_засыпки_м3",
        min_value=min(unique_df["Объем_засыпки_м3"]),
        max_value=max(unique_df["Объем_засыпки_м3"]),
    )
    Время_года = st.sidebar.selectbox("Время_года",
        (unique_df["Время_года"])
    )



    dict_data = {
        'Общее_время':Общее_время,
        'Количество_персонала': Количество_персонала,
        'Количество_оборудования':Количество_оборудования,
        'Количество_заявок_у_диспетчера':Количество_заявок_у_диспетчера,
        'Количество_диспетчеров':Количество_диспетчеров,
        'Количество_свободных_машин': Количеcтво_свободных_машин,
        'Количество_персонала_УБ':Количество_персонала_УБ,
        'Средняя_скорость_машин':Средняя_скорость_машин,
        'Время_остановок':Время_остановок,
        'Расстояние':Расстояние,
        'Опыт_мастера_ЛАЭС_лет':Опыт_мастера_ЛАЭС_лет,
        'Количество_выездов_на_участок':Количество_выездов_на_участок,
        'Опыт_экскаваторщика_лет':Опыт_экскаваторщика_лет,
        'Ручной_труд_тип_экскаватора':Ручной_труд_тип_экскаватора,
        'Глубина_вскрытия_котлована_м':Глубина_вcкрытия_котлована_м,
        'Классификация_грунта':Классификация_грунта,
        'Обводненность':Обводненность,
        'Количество_персонала_для_вскрытия':Количество_персонала_для_вскрытия,
        'Тип_изоляции':Тип_изоляции,
        'Количество_персонала1':Количество_персонала1,
        'Параметры_дефекта_длина':Параметры_дефекта_длина,
        'Параметры_дефекта_ширина':Параметры_дефекта_ширина,
        'Количество_работников':Количество_работников,
        'Длина_для_нанесения_изоляции':Длина_для_нанесения_изоляции,
        'Количество_персонала_':Количество_персонала_,
        'Объем_засыпки_м3':Объем_засыпки_м3,
        'Время_года':Время_года,
    }

    st.write(
        f"""### Данные процесса:\n
    'Общее_время':{dict_data['Общее_время']}
    'Количество_персонала': {dict_data['Количество_персонала']}
    'Количество_оборудования':{dict_data['Количество_оборудования']}
    'Количество_заявок_у_диспетчера':{dict_data['Количество_заявок_у_диспетчера']}
    'Количество_диспетчеров':{dict_data['Количество_диспетчеров']}
    'Количество_свободных_машин': {dict_data['Количество_свободных_машин']}
    'Количество_персонала_УБ':{dict_data['Количество_персонала_УБ']}
    'Средняя_скорость_машин':{dict_data['Средняя_скорость_машин']}
    'Время_остановок':{dict_data['Время_остановок']}
    'Расстояние':{dict_data['Расстояние']}
    'Опыт_мастера_ЛАЭС_лет':{dict_data['Опыт_мастера_ЛАЭС_лет']}
    'Количество_выездов_на_участок':{dict_data['Количество_выездов_на_участок']}
    'Опыт_экскаваторщика_лет':{dict_data['Опыт_экскаваторщика_лет']}
    'Ручной_труд_тип_экскаватора':{dict_data['Ручной_труд_тип_экскаватора']}
    'Глубина_вскрытия_котлована_м':{dict_data['Глубина_вскрытия_котлована_м']}
    'Классификация_грунта':{dict_data['Классификация_грунта']}
    'Обводненность':{dict_data['Обводненность']}
    'Количество_персонала_для_вскрытия':{dict_data['Количество_персонала_для_вскрытия']}
    'Тип_изоляции':{dict_data['Тип_изоляции']}
    'Количество_персонала1':{dict_data['Количество_персонала1']}
    'Параметры_дефекта_длина':{dict_data['Параметры_дефекта_длина']}
    'Параметры_дефекта_ширина':{dict_data['Параметры_дефекта_ширина']}
    'Количество_работников':{dict_data['Количество_работников']}
    'Длина_для_нанесения_изоляции':{dict_data['Длина_для_нанесения_изоляции']}
    'Количество_персонала_':{dict_data['Количество_персонала_']}
    'Объем_засыпки_м3':{dict_data['Объем_засыпки_м3']}
    'Время_года':{dict_data['Время_года']}
    """
    )

    # evaluate and return prediction (text)
    button_ok = st.button("Predict")
    if button_ok:
        predictions = requests.post(endpoint, timeout=8000, json=dict_data)
        json_str = json.dumps(predictions.json())
        output = json.loads(json_str)
        st.write(f"## Расчетное время: {(round(output, 2))}")


    with open("C:/Users/User/Desktop/ddk/config/params.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        real_predict_eval=pd.read_csv(config['preprocessing']['real_predict_eval'])

    real_eval={'real': dict_data['Общее_время'], 'eval': output}
    new_row = pd.DataFrame([real_eval])
    real_predict_eval = pd.concat([real_predict_eval, new_row], ignore_index=True)
    real_predict_eval.to_csv(config['preprocessing']['real_predict_eval'])






    fig = plt.figure(figsize=(2, 7))
    x = np.arange(real_predict_eval.shape[0])
    y = real_predict_eval['eval'] - real_predict_eval['real']
    for i in range(len(y)):
        if y.iloc[i] > real_predict_eval['real'].iloc[i] * 0.3:
             plt.plot(x[i], y.iloc[i], color='blue', marker='o')
        elif y.iloc[i] <-1*real_predict_eval['real'].iloc[i] * 0.3:
             plt.plot(x[i], y.iloc[i], color='red', marker='*')
        else:
            plt.plot(x[i], y.iloc[i], color='green', marker="P")
    plt.plot(x, real_predict_eval['real'] * 0.3, color='red', linestyle='dashed')
    plt.plot(x, -real_predict_eval['real'] * 0.3, color='red', linestyle='dashed')
    plt.plot(x, np.zeros(x.shape))
    plt.grid(visible=True)
    plt.xlabel('n_iter')
    plt.ylabel('Дельта')
    st.plotly_chart(fig, use_container_width=True)


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files:
    """
    button_ok = st.button("Predict")
    if button_ok:
        # заглушка так как не выводим все предсказания
        data_ = data[:5]
        output = requests.post(endpoint, files=files, timeout=8000)
        data_["predict"] = output.json()["prediction"]
        st.write(data_.head())

