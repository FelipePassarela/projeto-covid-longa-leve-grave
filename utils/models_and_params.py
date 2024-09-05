from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def svm_model_and_params():
    return SVC(random_state=42, probability=True), {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [2, 3, 4],
        "gamma": ["scale", "auto"]
    }


def lr_model_and_params():
    return LogisticRegression(random_state=42), {
        "C": [0.1, 1, 10],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    }


def knn_model_and_params():
    return KNeighborsClassifier(), {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
    }


def rf_model_and_params():
    return RandomForestClassifier(random_state=42), {
        "n_estimators": [100, 200, 300],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20, 30],
    }


def xgb_model_and_params():
    return XGBClassifier(random_state=42), {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.1, 0.01, 0.001],
        "subsample": [0.5, 0.7, 1],
        "colsample_bytree": [0.5, 0.7, 1],
    }
