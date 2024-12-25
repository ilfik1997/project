"""
Программа: Модель для прогнозирования того, сколько по времени должен проходить ДДК, основываясь на вписанных параметрах
Версия: 1.0
"""

import warnings
import pandas as pd

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel

from Backend.src.pipeline.pipeline_training import pipeline_training
from Backend.src.Evaluate.evaluate import pipeline_evaluate
from Backend.src.Train.get_metrics import  load_metrics

warnings.filterwarnings("ignore")

app = FastAPI()
CONFIG_PATH = "C:/Users/User/Desktop/ddk/config/params.yml"


class InsuranceCustomer(BaseModel):
    """
    Признаки для получения результатов модели
    """
    Количество_персонала: int
    Количество_оборудования: int
    Количество_заявок_у_диспетчера: int
    Количество_диспетчеров: int
    Количество_свободных_машин: int
    Количество_персонала_УБ: int
    Средняя_скорость_машин: int
    Время_остановок: int
    Расстояние: int
    Опыт_мастера_ЛАЭС_лет: int
    Количество_выездов_на_участок: int
    Опыт_экскаваторщика_лет: int
    Ручной_труд_тип_экскаватора: str
    Глубина_вскрытия_котлована_м: float
    Классификация_грунта: str
    Обводненность: str
    Количество_персонала_для_вскрытия: int
    Тип_изоляции: str
    Количество_персонала1: int
    Параметры_дефекта_длина: float
    Параметры_дефекта_ширина: float
    Количество_работников: int
    Длина_для_нанесения_изоляции: float
    Количество_персонала_: int
    Объем_засыпки_м3: float
    Время_года: str




@app.get("/hello")
def welcome():
    """
    Hello
    :return: None
    """
    return {'message': 'Hello Data Scientist!'}


@app.post("/train")
def training():
    """
    Обучение модели, логирование метрик
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {"metrics": metrics}


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), "Результат не соответствует типу list"
    # заглушка так как не выводим все предсказания, иначе зависнет
    return {"prediction": result[:5]}


@app.post("/predict_input")
def prediction_input(customer: InsuranceCustomer):
    """
    Предсказание модели по введенным данным
    """
    features = [
        [
            customer.Количество_персонала,
            customer.Количество_оборудования,
            customer.Количество_заявок_у_диспетчера,
            customer.Количество_диспетчеров,
            customer.Количество_свободных_машин,
            customer.Количество_персонала_УБ,
            customer.Средняя_скорость_машин,
            customer.Время_остановок,
            customer.Расстояние,
            customer.Опыт_мастера_ЛАЭС_лет,
            customer.Количество_выездов_на_участок,
            customer.Опыт_экскаваторщика_лет,
            customer.Ручной_труд_тип_экскаватора,
            customer.Глубина_вскрытия_котлована_м,
            customer.Классификация_грунта,
            customer.Обводненность,
            customer.Количество_персонала_для_вскрытия,
            customer.Тип_изоляции,
            customer.Количество_персонала1,
            customer.Параметры_дефекта_длина,
            customer.Параметры_дефекта_ширина,
            customer.Количество_работников,
            customer.Длина_для_нанесения_изоляции,
            customer.Количество_персонала_,
            customer.Объем_засыпки_м3,
            customer.Время_года
        ]
    ]

    cols = [
            'Количество_персонала',
            'Количество_оборудования',
            'Количество_заявок_у_диспетчера',
            'Количество_диспетчеров',
            'Количество_свободных_машин',
            'Количество_персонала_УБ',
            'Средняя_скорость_машин',
            'Время_остановок',
            'Расстояние',
            'Опыт_мастера_ЛАЭС_лет',
            'Количество_выездов_на_участок',
            'Опыт_экскаваторщика_лет',
            'Ручной_труд_тип_экскаватора',
            'Глубина_вскрытия_котлована_м',
            'Классификация_грунта',
            'Обводненность',
            'Количество_персонала_для_вскрытия',
            'Тип_изоляции',
            'Количество_персонала1',
            'Параметры_дефекта_длина',
            'Параметры_дефекта_ширина',
            'Количество_работников',
            'Длина_для_нанесения_изоляции',
            'Количество_персонала_',
            'Объем_засыпки_м3',
            'Время_года'
    ]

    data = pd.DataFrame(features, columns=cols)
    predictions = pipeline_evaluate(config_path=CONFIG_PATH, dataset=data)[0]

    return predictions



if __name__ == "__main__":
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host="127.0.0.1", port=80)