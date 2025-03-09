import os
import pickle
from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier

from utils.models.evaluate_models import extract_subset
from utils.models.models_and_params import get_model_name


def _calculate_shap_values(
        model: BaseEstimator,
        X_train_selected: np.ndarray,
        X_test_selected: np.ndarray
    ) -> np.ndarray:
    """
    Calculate SHAP values for the given model and data.

    :param model: The trained model to explain
    :param X_train_selected: Training data used for background distribution
    :param X_test_selected: Test data to explain

    :return: SHAP values for the test data
    """
    if isinstance(model, XGBClassifier):
        shap_values = shap.TreeExplainer(model).shap_values(X_test_selected)
    elif isinstance(model, RandomForestClassifier):
        shap_values = shap.TreeExplainer(model).shap_values(X_test_selected)
        shap_values = np.array(shap_values)[:, :, 1]
    else:
        background = shap.sample(X_train_selected, 100)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_test_selected)
    return shap_values


def _plot_shap(
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        X_columns: pd.Index,
        model: BaseEstimator,
        selector: RFE,
        n_feats: int,
        save_path: PathLike
    ) -> None:
    """
    Plot the SHAP values for the given model and number of features.

    :param X_train: The training data.
    :param X_test: The testing data.
    :param X_columns: The columns of the data.
    :param model: The trained model to explain.
    :param selector: The feature selector used to select the features.
    :param n_feats: Number of features to select.
    :param save_path: Path to the directory where the plots will be saved.
    """
    X_train_selected, _ = extract_subset(selector, X_train, n_feats)
    X_test_selected, feat_indices = extract_subset(selector, X_test, n_feats)
    shap_values = _calculate_shap_values(model, X_train_selected, X_test_selected)

    features_names = X_columns[feat_indices]
    shap.summary_plot(shap_values, X_test_selected, feature_names=features_names, show=False)
    plt.title(f"SHAP values of the {get_model_name(model, short=True)} model")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / f"{n_feats}feats.png")
    plt.clf()
    plt.close()


def _get_best_model(
        n_feats: int,
        models_path: PathLike,
        results_path: PathLike,
        comparison_metric: str = "roc_auc"
    ) -> tuple[str, float]:
    """
    Get the best performing model and its score for a specific number of features.

    :param n_feats: Number of features the models were trained with.
    :param models_path: Path to the directory containing the model files.
    :param results_path: Path to the directory containing the results files.
    :param comparison_metric: Metric to use for comparing models. Defaults to "roc_auc".

    :return: Tuple containing (best_model_name, best_score).
    """
    models_names = os.listdir(models_path)
    models_names = [model for model in models_names if not model.startswith("RFE")]
    models_names = [model for model in models_names if model.endswith(f"_{n_feats}feats.pkl")]

    best_score = 0
    best_model = ""
    for name in models_names:
        name = name.split("_")[0]
        results = pd.read_csv(f"{results_path}/{name}.csv")
        score = results.loc[results["n_features"] == n_feats, comparison_metric].values[0]
        if score > best_score:
            best_score = score
            best_model = name

    return best_model, best_score


def plot_shaps(
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        X_columns: pd.Index,
        selector: RFE,
        features_array: list[int],
        models_path: str,
        save_path: PathLike,
        specific_model: BaseEstimator = None,
    ) -> None:
    """
    Plot the best model's SHAP values for different number of features.

    Load the selectors from the disk and perform same transformations as in the evaluate_models.py script. Then, calculate
    the SHAP values for the best model and plot them.

    :param X_train: The training data.
    :param X_test: The testing data.
    :param X_columns: The columns of the data.
    :param models_path: Path to the directory containing the model files.
    :param save_path: Path to the directory where the plots will be saved.
    :param specific_model: Specific model to plot the SHAP values for. If
        provided, the best model will not be selected. Defaults to None.
    """
    print("Plotting SHAP values...")

    for n_feats in features_array:
        trained_model_path = _get_trained_model_path(models_path, n_feats, specific_model=specific_model)

        with open(trained_model_path, "rb") as f:
            model = pickle.load(f)
        print(f"{get_model_name(model, short=True)} - {n_feats} features")

        _plot_shap(X_train, X_test, X_columns, model, selector, n_feats, save_path)


def _plot_shap_feature_clustering(
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        X_columns: pd.Index,
        model: BaseEstimator,
        selector: RFE,
        n_feats: int,
        save_path: PathLike
    ) -> None:
    """
    Plot the SHAP values for the given model and number of features.

    :param X_train: The training data.
    :param X_test: The testing data.
    :param y_test: The testing labels.
    :param X_columns: The columns of the data.
    :param model: The trained model to explain.
    :param selector: The feature selector used to select the features.
    :param n_feats: Number of features to select.
    :param save_path: Path to the directory where the plots will be saved.
    """
    X_train_selected, _ = extract_subset(selector, X_train, n_feats)
    X_test_selected, feat_indices = extract_subset(selector, X_test, n_feats)
    shap_values = _calculate_shap_values(model, X_train_selected, X_test_selected)

    features_names = X_columns[feat_indices]
    clustering = shap.utils.hclust(X_test_selected, y_test)
    shap.bar_plot(shap_values, clustering=clustering, feature_names=features_names, show=False)
    plt.title(f"SHAP values of the {get_model_name(model, short=True)} model")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / f"{n_feats}feats.png")
    plt.clf()
    plt.close()


def plot_shaps_feature_clustering(
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        X_columns: pd.Index,
        selector: RFE,
        features_array: list[int],
        models_path: PathLike,
        save_path: PathLike,
        specific_model: BaseEstimator = None
    ) -> None:
    """
    Plot the best model's SHAP values for different number of features.

    Load the selectors from the disk and perform same transformations as in the evaluate_models.py script. Then, calculate
    the SHAP values for the best model and plot them.

    :param X_train: The training data.
    :param X_test: The testing data.
    :param y_test: The testing labels.
    :param X_columns: The columns of the data.
    :param models_path: Path to the directory containing the model files.
    :param save_path: Path to the directory where the plots will be saved.
    :param specific_model: Specific model to plot the SHAP values for. If
        provided, the best model will not be selected. Defaults to None.
    """
    print("Plotting SHAP values with clustering...")

    for n_feats in features_array:
        trained_model_path = _get_trained_model_path(models_path, n_feats, specific_model=specific_model)

        with open(trained_model_path, "rb") as f:
            model = pickle.load(f)
        print(f"{get_model_name(model, short=True)} - {n_feats} features")

        _plot_shap_feature_clustering(X_train, X_test, y_test, X_columns, model, selector, n_feats, save_path)


def _get_trained_model_path(
        models_path: PathLike, 
        n_feats: int,
        specific_model: BaseEstimator = None
    ) -> Path:
    """
    Get the path to the trained model.

    :param models_path: Path to the directory containing the model files.
    :param n_feats: Number of features the models were trained with.
    :param specific_model: Specific model to plot the SHAP values for. If
        provided, the best model will not be selected.

    :return: Path to the trained model file.
    """
    if specific_model:
        model_name = get_model_name(specific_model, short=True)
    else:
        std_models_path = Path(models_path) / "standard"
        results_path = Path("output/results/test/standard/")
        model_name, _ = _get_best_model(n_feats, std_models_path, results_path)

    trained_model_path = Path(models_path) / "standard" / f"{model_name}_{n_feats}feats.pkl"
    return trained_model_path