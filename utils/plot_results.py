import os
import pickle
from os import PathLike
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from xgboost import XGBClassifier

from utils.evaluate_models import EvalResultsDict, extract_subset
from utils.models_and_params import get_model_name

EvalSummaryDict = Dict[str, Dict[str, pd.DataFrame]]

SCORE_TITLES = {
    "accuracy": "Accuracy",
    "f1"      : "F1 Score",
    "roc_auc" : "ROC AUC",
}

RESULTS_PATHS = {
    'test_standard' : 'output/results/test/standard',
    'train_standard': 'output/results/train/standard',
    'test_tuned'    : 'output/results/test/tuned',
    'train_tuned'   : 'output/results/train/tuned',
}

SUBPLOT_TITLES = {
    'test_standard' : 'Test (Standard)',
    'train_standard': 'Train (Standard)',
    'test_tuned'    : 'Test (Tuned)',
    'train_tuned'   : 'Train (Tuned)',
}


def plot_result(
        ax: plt.Axes,
        results_dict: Dict[str, pd.DataFrame], 
        score: str,
        title: str = None,
        subtitle: str = None,
        path: PathLike = None
    ) -> None:
    """
    Plot the results of different models for different number of features.

    :param ax: The axis to plot the results.
    :param results_dict: Dictionary containing the results for each model.
    :param score: Score to be plotted. Should be one of the score columns in the CSV files.
    :param title: Title to be added to the plot. Defaults to None.
    :param subtitle: Subtitle to be added to the plot. Defaults to None.
    :param path: If provided, the plot will be saved to this path.
    """
    for model_name, results in results_dict.items():
        ax.plot(results["n_features"], results[score], label=model_name, marker='o')

    ax_title = title if title else f"Model Comparison ({SCORE_TITLES[score]})"
    ax_title += f"\n{subtitle}" if subtitle else ""
    n_features = next(iter(results_dict.values()))["n_features"]
    
    ax.set_title(f"{ax_title}")
    ax.set_xticks(n_features)
    ax.set_xlabel("Number of SNPs", fontsize=12)
    ax.set_ylabel(SCORE_TITLES[score], fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--')

    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = ax.get_figure()
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def plot_eval_summary(
        mode_results: EvalSummaryDict,
        score: str,
        path: PathLike
    ) -> None:
    """
    Plot the summary of the evaluation results.

    :param mode_results: Dictionary containing the results for each mode (train/test, standard/tuned).
    :param score: Score to be plotted. Should be one of the score columns in the CSV files.
    :param path: Path to save the plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    fig.suptitle("Standard vs Tuned Comparison (Test and Train)", fontsize=18)
    
    subtitles = list(mode_results.keys())
    flat_axes = axes.flatten()
    for ax, subtitle in zip(flat_axes, subtitles):
        plot_result(ax, mode_results[subtitle], score, title=subtitle, subtitle=None)
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    fig.text(0.5, 0.02, "Number of SNPs", ha='center', va='center', fontsize=14)
    fig.text(0.03, 0.5, SCORE_TITLES[score], ha='center', va='center', rotation='vertical', fontsize=14)
    plt.tight_layout(rect=[0.03, 0.03, 1, 1])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def plot_shap(
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        X_columns: pd.Index,
        selector: RFE,
        features_array: list[int],
        models_path: str,
        comparison_metric: str = "roc_auc"
    ) -> None:
    """
    Plot the best model's SHAP values for different number of features.

    Load the selectors from the disk and perform same transformations as in the evaluate_models.py script. Then, calculate
    the SHAP values for the best model and plot them.

    :param X_train: The training data.
    :param X_test: The testing data.
    :param X_columns: The columns of the data.
    :param models_path: Path to the directory containing the model files.
    :param comparison_metric: Metric to use for comparing models. Defaults to "roc_auc".
    """
    # TODO: Refactor this function to uncouple the directory handling from the actual plotting logic.
    print("Plotting SHAP values...")
    for n_feats in features_array:
        path = Path(models_path) / "standard"
        results_path = "output/results/test/standard/"
        model_standard, _ = get_best_model(n_feats, path, results_path, comparison_metric)
        model_path = f"standard/{model_standard}_{n_feats}feats.pkl"

        with open(f"{models_path}/{model_path}", "rb") as f:
            model = pickle.load(f)
        print(f"{get_model_name(model, short=True)} - {n_feats} features")

        X_train_selected, _ = extract_subset(selector, X_train, n_feats)
        X_test_selected, feat_indices = extract_subset(selector, X_test, n_feats)
        shap_values = calculate_shap_values(model, X_train_selected, X_test_selected)

        features_names = X_columns[feat_indices]
        shap.summary_plot(shap_values, X_test_selected, feature_names=features_names, show=False)
        plt.title(f"SHAP values of the {get_model_name(model, short=True)} model")
        plt.tight_layout()

        os.makedirs("output/plots/shap", exist_ok=True)
        plt.savefig(f"output/plots/shap/{n_feats}feats.png")
        plt.clf()
        plt.close()


def plot_shap_svm(
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        X_columns: pd.Index,
        selector: RFE,
        features_array: list[int],
        models_path: str,
        comparison_metric: str = "roc_auc"
    ) -> None:
    """
    Plot the best model's SHAP values for different number of features.

    Load the selectors from the disk and perform same transformations as in the evaluate_models.py script. Then, calculate
    the SHAP values for the best model and plot them.

    :param X_train: The training data.
    :param X_test: The testing data.
    :param X_columns: The columns of the data.
    :param models_path: Path to the directory containing the model files.
    :param comparison_metric: Metric to use for comparing models. Defaults to "roc_auc".
    """
    # TODO: Refactor this function to uncouple the directory handling from the actual plotting logic.
    print("Plotting SVM SHAP values...")
    for n_feats in features_array:
        model_name = get_model_name(SVC(), short=True)
        model_path = f"standard/{model_name}_{n_feats}feats.pkl"

        with open(f"{models_path}/{model_path}", "rb") as f:
            model = pickle.load(f)
        print(f"{get_model_name(model, short=True)} - {n_feats} features")

        X_train_selected, _ = extract_subset(selector, X_train, n_feats)
        X_test_selected, feat_indices = extract_subset(selector, X_test, n_feats)
        shap_values = calculate_shap_values(model, X_train_selected, X_test_selected)

        features_names = X_columns[feat_indices]
        shap.summary_plot(shap_values, X_test_selected, feature_names=features_names, show=False)
        plt.title(f"SHAP values of the {get_model_name(model, short=True)} model")
        plt.tight_layout()

        os.makedirs(f"output/plots/shap/svm", exist_ok=True)
        plt.savefig(f"output/plots/shap/svm/{n_feats}feats.png")
        plt.clf()
        plt.close()


def calculate_shap_values(
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


def get_best_model(
        n_feats: int, 
        models_path: str, 
        results_path: str, 
        comparison_metric: str = "roc_auc"
    ) -> tuple[str, float]:
    """
    Get the best performing model and its score for a specific number of features.

    :param n_feats: Number of features the models were trained with.
    :param models_path: Path to the directory containing the model files.
    :param results_path: Path to the directory containing the results files.
    :param comparison_metric: Metric to use for comparing models. Defaults to "roc_auc".

    :return: Tuple containing (best_model_filename, best_score)
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


def plot_evals(
    plots_path: PathLike, 
    results_standard: EvalResultsDict,
    results_tuned: EvalResultsDict,
    eval_metric: str
    ) -> None:
    """
    Plot the evaluation results for the standard and tuned models.

    :param plots_path: Path to the directory where the plots will be saved.
    :param results_standard: Dictionary containing the results for the standard models.
    :param results_tuned: Dictionary containing the results for the tuned models.
    :param eval_metric: Metric to be plotted. Should be one of the score columns in the CSV files.
    """
    eval_summary = {
        "Test (Standard)": {model_name: sets["test"] for model_name, sets in results_standard.items()},
        "Train (Standard)": {model_name: sets["train"] for model_name, sets in results_standard.items()},
        "Test (Tuned)": {model_name: sets["test"] for model_name, sets in results_tuned.items()},
        "Train (Tuned)": {model_name: sets["train"] for model_name, sets in results_tuned.items()},
    }
    for eval_set, result in eval_summary.items():
        fig_filename = f"{eval_set}.png".lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = plots_path / fig_filename
        ax = plt.gca()
        plot_result(ax, result, eval_metric, "ROC AUC of Different Models", eval_set, path)

    path = plots_path / "summary.png"
    plot_eval_summary(eval_summary, eval_metric, path)


def plot_boxplot(
        results_df: pd.DataFrame, 
        model_name: str, 
        plots_path: PathLike, 
        scoring="roc_auc",
    ) -> plt.Figure:
    """
    Plot the boxplot of the results for different number of features.

    :param results_df: DataFrame containing the results for different number of features.
    :param model_name: Name of the model.
    :param plots_path: Path to the directory where the plots will be saved.
    :param scoring: Metric to be plotted. Defaults to "roc_auc".

    :return: The figure containing the boxplot.
    """

    scores_per_feature = {n: scores for n, scores in zip(results_df["n_features"], results_df["scores"])}
    score_title = SCORE_TITLES.get(scoring, scoring)

    cv = results_df["scores"].apply(len).max()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(list(scores_per_feature.values()), tick_labels=list(scores_per_feature.keys()))
    ax.set_xlabel("Number of SNPs", fontsize=12)
    ax.set_ylabel(score_title, fontsize=12)
    ax.set_title(f"Distribution of {score_title} for {model_name} ({cv}-fold CV)", fontsize=14)
    plt.tight_layout()

    fig_path = Path(plots_path) / f"boxplot_{model_name}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)

    return fig