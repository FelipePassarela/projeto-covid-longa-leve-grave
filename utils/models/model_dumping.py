import os
import pickle
from os import PathLike
from pathlib import Path

from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFE, SelectorMixin
from sklearn.svm import SVC

from utils.models.models_and_params import get_model_name


def load_rfe_selector(n_features: int, path: PathLike) -> RFE:
    """
    Load the Recursive Feature Elimination (RFE) selector with a specific number of features.

    If the selector does not exist in the disk, create a new one.

    :param n_features: The number of features to be selected.
    :param path: The path to the folder where the selector is saved.

    :return: The RFE selector or None if it does not exist.
    """
    model_file = Path(path) / f"RFE_{n_features}feats.pkl"
    try:
        with open(model_file, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None


def save_model(
        model: BaseEstimator | SelectorMixin,
        n_features: int, 
        path: PathLike
    ) -> None:
    """
    Save the model in the disk.

    :param model: The model to be saved.
    :param n_features: The number of features used to train the model.
    :param path: The path to the folder where the model will be saved.
    """
    os.makedirs(path, exist_ok=True)
    model_name = get_model_name(model, short=True)
    model_file = Path(f"{path}/{model_name}_{n_features}feats.pkl")
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)
