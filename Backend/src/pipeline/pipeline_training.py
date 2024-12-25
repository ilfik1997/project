import yaml
from Backend.src.Data.split_df import get_train_test_data, split_train_test,split_val
from Backend.src.Data.get_data import get_dataset
from Backend.src.Train.Train import grid_search, train_model
from Backend.src.Transform.Transform import pipeline_preprocess
import joblib
import os



def pipeline_training(config_path: str) -> None:
    """
    Полный цикл получения данных, предобработки и тренировки модели
    :param config_path: путь до файла с конфигурациями
    :return: None
    """
    # get params
    with open(config_path,encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preproc = config["preprocessing"]
    training = config["train"]

    # get data
    train_data =get_dataset(preproc['train_path']).drop(columns=preproc["drop_columns"], axis=1, errors="ignore")

    # preprocessing
    train_data = pipeline_preprocess(data=train_data, flg_evaluate=False, **preproc)

    # split data
    df_train, df_test = split_train_test(dataset=train_data, **preproc)

    #get x_test/train, y_test,train
    X_train, X_test, y_train, y_test =  get_train_test_data(df_train, df_test,target=preproc['target_column'])

    #get val data
    X_train_, X_val, y_train_, y_val= split_val(X_train, y_train)


    # find optimal params
    grid_result = grid_search(y_train, X_train, X_val)

    # train with optimal params
    cat_grid=train_model(X_test,y_test,X_train_,y_train_,X_val,y_val,metric_path=training['metrics_path'],
                         grid_search_result=grid_result)

    # save result (study, model)
    joblib.dump(cat_grid, os.path.join(training["model_path"]))
    joblib.dump(grid_result, os.path.join(training["study_path"]))




