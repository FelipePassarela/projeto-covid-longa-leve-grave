from typing import Any, Dict, List, Tuple

from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

HyperParamGrid = Dict[str, List[Any]]  # Type alias for hyperparameters grid.


def get_model_and_params(model_name: str) -> Tuple[BaseEstimator, HyperParamGrid]:
    """
    Return the model and the hyperparameters to be tested in the RandomizedSearchCV based on the model name.

    :param model_name: The name of the model.
    
    :return: The model and the hyperparameters.
    """

    models_params = {
        "svm": (
            SVC(random_state=42, probability=True), 
            { 
                "C": uniform(0.1, 10),
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "degree": randint(2, 5),
                "gamma": ["scale", "auto"]
            }
        ),
        "logistic_regression": (
            LogisticRegression(random_state=42),
            {
                "C": uniform(0.1, 10),
                "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
            }
        ),
        "knn": (
            KNeighborsClassifier(),
            {
                "n_neighbors": randint(3, 10),
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
            }
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": randint(100, 301),
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 10, 20, 30],
            }
        ),
        "xgboost": (
            XGBClassifier(random_state=42),
            {
                "n_estimators": randint(100, 301),
                "max_depth": randint(3, 6),
                "learning_rate": uniform(0.001, 0.1),
                "subsample": uniform(0.5, 0.5),
                "colsample_bytree": uniform(0.5, 0.5),
            }
        )
    }

    if model_name in models_params:
        return models_params[model_name]
    else:
        raise ValueError(f"Model '{model_name}' is not supported. Choose from {list(models_params.keys())}.")
    

def get_model_name(model: BaseEstimator | RFE | str, short: bool = False) -> str:
    """
    Return the model name based on the model object.

    :param model: The model object.
    :param short: Whether to return the short name or not.
    
    :return: The model name.
    """
    
    long_names = {
        LogisticRegression: "Logistic Regression",
        SVC: "Support Vector Machine",
        KNeighborsClassifier: "K-Nearest Neighbors",
        RandomForestClassifier: "Random Forest",
        XGBClassifier: "XGBoost",
        RFE: "Recursive Feature Elimination"
    }

    short_names = {
        LogisticRegression: "LR",
        SVC: "SVM",
        KNeighborsClassifier: "KNN",
        RandomForestClassifier: "RF",
        XGBClassifier: "XGB",
        RFE: "RFE"
    }

    names = short_names if short else long_names

    for model_class, name in names.items():
        if isinstance(model, model_class):
            return name
        if isinstance(model, str) and model == model_class.__name__:
            return names[model_class]
        
    raise ValueError(f"Model '{model}' is not supported. Choose from {list(long_names.values())}.")
