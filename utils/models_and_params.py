from typing import Dict, List, Tuple, Union
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


HyperParamValue = Union[int, float, str]           # Type alias for hyperparameter values.
HyperParamGrid = Dict[str, List[HyperParamValue]]  # Type alias for hyperparameters grid.

MODELS_AND_PARAMS = {
    "svm": (
        SVC(random_state=42, probability=True),
        {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": [2, 3, 4],
            "gamma": ["scale", "auto"]
        }
    ),
    "logistic_regression": (
        LogisticRegression(random_state=42),
        {
            "C": [0.1, 1, 10],
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
        }
    ),
    "knn": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
        }
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [100, 200, 300],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20, 30],
        }
    ),
    "xgboost": (
        XGBClassifier(random_state=42),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.1, 0.01, 0.001],
            "subsample": [0.5, 0.7, 1],
            "colsample_bytree": [0.5, 0.7, 1],
        }
    )
}


def get_model_and_params(model_name: str) -> Tuple[BaseEstimator, HyperParamGrid]:
    """
    Return the model and the hyperparameters to be tested in the GridSearchCV based on the model name.

    :param model_name: The name of the model.
    :type model_name: str
    
    :return: The model and the hyperparameters.
    :rtype: Tuple[BaseEstimator, Dict[str, List[HyperParamValue]]]
    """
    
    if model_name in MODELS_AND_PARAMS:
        return MODELS_AND_PARAMS[model_name]
    else:
        raise ValueError(f"Model '{model_name}' is not supported. Choose from {list(MODELS_AND_PARAMS.keys())}.")