from catboost import CatBoostRegressor

import numpy as np

from Backend.src.Train.get_metrics import save_metrics


def grid_search(y_train, X_train, X_val):
    cat_features = X_val.select_dtypes('category').columns.tolist()
    cat_features.append('Обводненность')

    grid = {
        "learning_rate": np.logspace(-3, -1, 3),
        "boosting_type": ['Ordered', 'Plain'],
        "max_depth": list(range(3, 12)),
        "l2_leaf_reg": np.logspace(-5, 2, 5),
        "bootstrap_type": ["Bayesian", "Bernoulli", "MVS", "No"],
        'border_count': [128, 254],
        'grow_policy': ["SymmetricTree", "Depthwise", "Lossguide"],
        "random_state": [10]

    }
    model = CatBoostRegressor(loss_function="MAE",
                              eval_metric="MAE",
                              cat_features=cat_features,
                              silent=True)
    grid_search_result = model.randomized_search(
        grid,
        X=X_train,
        y=y_train,
        n_iter=10)

    return grid_search_result['params']

def train_model(X_test, y_test, X_train_, y_train_, X_val, y_val, metric_path: str,
                grid_search_result=None):
    cat_features = X_val.select_dtypes('category').columns.tolist()
    cat_features.append('Обводненность')
    eval_set = [(X_val, y_val)]

    cat_grid = CatBoostRegressor(**grid_search_result,
                                 loss_function='MAE',
                                 eval_metric='MAE')
    cat_grid.fit(X_train_,
                 y_train_,
                 cat_features=cat_features,
                 eval_set=eval_set,
                 verbose=False,
                 early_stopping_rounds=100)
    save_metrics(X_test=X_test, y_test=y_test, model=cat_grid, metric_path=metric_path)
    return cat_grid
