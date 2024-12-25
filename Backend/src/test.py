import warnings
import pandas as pd
import os
import yaml
import joblib
import pandas as pd
from Backend.src.Data.get_data import  get_dataset
from Backend.src.Transform.Transform import pipeline_preprocess

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel

from Backend.src.pipeline.pipeline_training import pipeline_training
from Backend.src.Evaluate.evaluate import pipeline_evaluate
from Backend.src.Train.get_metrics import  load_metrics
import json

warnings.filterwarnings("ignore")

app = FastAPI()
CONFIG_PATH = "C:/Users/User/Desktop/ddk/config/params.yml"

pipeline_training(config_path=CONFIG_PATH)








