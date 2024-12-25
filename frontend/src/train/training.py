"""
Программа: Тренировка модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import os
import json
import joblib
import requests
import streamlit as st


def start_training(config: dict, endpoint: object) -> None:
    """
    Тренировка модели с выводом результатов
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """
    # Last metrics
    if os.path.exists(config["train"]["metrics_path"]):
        with open(config["train"]["metrics_path"]) as json_file:
            old_metrics = json.load(json_file)
    else:
        # если до этого не обучали модель и нет прошлых значений метрик
        old_metrics = {"MAE": 0, "MSE": 0, "RMSE": 0,
                       "RMSLE": 0, "R2 adjusted": 0, "MPE_%": 0,
                       "MAPE_%": 0, "WAPE_%": 0}

        # Train
    with st.spinner("Модель подбирает параметры..."):
        output = requests.post(endpoint, timeout=8000)
    st.success("Success!")

    new_metrics = output.json()["metrics"]


    # diff metrics
    st.markdown(
        """
        <style>
        .custom-metric {
            font-size: 36px; /* Change this to your desired size */
            color: #1D7F1D; /* Change the color if needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    MAE, MSE, RMSE, RMSLE, R2_adjusted, MPE, MAPE, WAPE = st.columns(8)
    MAE.metric(
        "MAE",
        new_metrics["MAE"],
        f"{new_metrics['MAE']-old_metrics['MAE']:.3f}",

    )
    MSE.metric(
        "MSE",
        new_metrics["MSE"],

        f"{new_metrics['MSE'] - old_metrics['MSE']:.3f}",
    )
    RMSE.metric(
        "RMSE",
        new_metrics["RMSE"],
        f"{new_metrics['RMSE'] - old_metrics['RMSE']:.3f}",
    )
    RMSLE.metric(
        "RMSLE",
        new_metrics["RMSLE"],
        f"{new_metrics['RMSLE'] - old_metrics['RMSLE']:.3f}",
    )
    R2_adjusted.metric(
        "R2 adjusted",
        new_metrics["R2 adjusted"],
        f"{new_metrics['R2 adjusted'] - old_metrics['R2 adjusted']:.3f}",
    )
    MPE.metric(
        "MPE_%",
        new_metrics["MPE_%"],
        f"{new_metrics['MPE_%'] - old_metrics['MPE_%']:.3f}",
    )
    MAPE.metric(
        "MAPE_%",
        new_metrics["MAPE_%"],
        f"{new_metrics['MAPE_%'] - old_metrics['MAPE_%']:.3f}",
    )
    WAPE.metric(
        "WAPE_%",
        new_metrics["WAPE_%"],
        f"{new_metrics['WAPE_%'] - old_metrics['WAPE_%']:.3f}",
    )



