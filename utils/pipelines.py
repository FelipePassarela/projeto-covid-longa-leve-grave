import os
import shutil
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectorMixin
from sklearn.model_selection import train_test_split

from utils.models.evaluate_models import (EvalResultsDict, evaluate_cv,
                                          evaluate_models)
from utils.models.model_dumping import load_rfe_selector, save_model
from utils.models.models_and_params import get_model_and_params, get_model_name
from utils.plots.results import plot_boxplot, plot_evals
from utils.plots.shap import plot_shaps, plot_shaps_bar_plot
from utils.preprocessing import fit_selector, preprocess_data


def main_pipeline(
        genomic_data_path: PathLike, 
        target: str,
        features_array: list[int],
        oversample: bool = False,
        selector_estim: BaseEstimator = RandomForestClassifier(random_state=42, n_jobs=-1),
        fit_selector_on_whole_dataset: bool = False,
        missing_threshold: float = 10.0,
        run_cv: bool = True,
        specific_model_for_shaps: BaseEstimator = None,
        eval_metric: str = "roc_auc",
        plot_shap: bool = True,
        plot_bar: bool = False
    ) -> None:
    """
    Executes the main pipeline of the project.

    Loads genomic data, preprocesses it, trains various machine learning models with different 
    feature sets, and evaluates their performance. Also generates visualization plots and
    SHAP value analysis.

    The workflow includes:
    1. Loading and preprocessing the data
    2. Feature selection using RFE
    3. Training and evaluating multiple models
    4. Generating performance plots and analysis visualizations

    :param genomic_data_path: Path to the genomic data file.
    :param target: The target variable for the dataset.
    :param features_array: List of number of features to be selected.
    :param oversample: Whether to oversample the data. Default is False.
    :param selector_estim: The estimator to be used for feature selection. Default is RandomForestClassifier.
    :param fit_selector_on_whole_dataset: Whether to fit the selector on the whole dataset. Default is False.
    :param missing_threshold: Maximum percentage of missing values allowed for a column to be kept. Default is 10.0.
    :param run_cv: Whether to run cross-validation. Default is True.
    :param specific_model_for_shaps: A specific model to generate SHAP plots for. If None,
        only the plots for the best model will be generated. Default is None.
    :param eval_metric: The evaluation metric to use. Default is "roc_auc".
    :param plot_shap: Whether to plot the SHAP values. Default is True.
    :param plot_bar: Whether to plot the SHAP bar plot. Default is False.
    """
    models_path = Path("output/models/")
    results_path = Path("output/results/")
    plots_path = Path("output/plots/")
    shap_path = plots_path / "shap"
    selectors_path = models_path / "selectors"

    X, X_train, X_test, y_train, y_test = data_preparing_pipeline(genomic_data_path, target, oversample, missing_threshold)

    selector = load_rfe_selector(1, selectors_path)
    if selector is None:
        selector = RFE(selector_estim, n_features_to_select=1, step=1, verbose=2)
        fit_selector(
            X_train, X_test, y_train, y_test, 
            selector, on_whole_dataset=fit_selector_on_whole_dataset
        )
    save_model(selector, 1, selectors_path)

    models_and_params = [
        get_model_and_params("logistic_regression"),
        get_model_and_params("svm"),
        get_model_and_params("knn"),
        get_model_and_params("random_forest"),
        get_model_and_params("xgboost")
    ]

    results_standard = evaluate_models(
        X_train, X_test, y_train, y_test, X.columns,
        selector, features_array, models_and_params,
        models_path, results_path, tune=False
    )
    results_tuned = evaluate_models(
        X_train, X_test, y_train, y_test, X.columns,
        selector, features_array, models_and_params,
        models_path, results_path, tune=True
    )

    if run_cv:
        model_cv = selector_estim
        results_cv = evaluate_cv(
            np.concatenate([X_train, X_test]),
            np.concatenate([y_train, y_test]),
            model=model_cv,
            selector=selector,
            features_array=features_array,
            results_path=results_path,
            scoring=eval_metric,
            cv=5,
            fitted_on_whole_dataset=fit_selector_on_whole_dataset
        )
    else:
        model_cv = None
        results_cv = None

    _plots_pipeline(
        X, X_train, X_test, y_test,
        selector, features_array,
        models_path, plots_path, shap_path,
        results_standard, results_tuned, eval_metric,
        model_cv=model_cv, results_cv=results_cv,
        specific_model_for_shaps=specific_model_for_shaps,
        plot_shap=plot_shap, plot_bar=plot_bar
    )


def data_preparing_pipeline(
        genomic_data_path: PathLike, 
        target: str, 
        oversample: bool = False, 
        missing_threshold: float = 10.0
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares the data for the pipeline by loading, preprocessing, and splitting it.
    
    :param genomic_data_path: Path to the genomic data file.
    :param target: The target variable for the dataset.
    :param oversample: Whether to oversample the data.
    :param missing_threshold: Maximum percentage of missing values allowed for a column to be kept.

    :return: tuple like (X, X_train, X_test, y_train, y_test)
    """
    df = pd.read_csv(genomic_data_path)
    df = df.dropna(subset=[target])
    missing_percentage = df.isnull().mean() * 100
    df = df.drop(columns=missing_percentage[missing_percentage > missing_threshold].index)
    df = df.drop(columns=["id"])

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test, oversample=oversample)

    X_columns = X.columns
    X = np.vstack([X_train, X_test])
    X = pd.DataFrame(X, columns=X_columns)

    return X, X_train, X_test, y_train, y_test


def _plots_pipeline(
        X: pd.DataFrame | np.ndarray,
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        selector: SelectorMixin,
        features_array: list,
        models_path: PathLike,
        plots_path: PathLike,
        shap_path: PathLike,
        results_standard: EvalResultsDict,
        results_tuned: EvalResultsDict,
        eval_metric: str,
        model_cv: BaseEstimator = None,
        results_cv: pd.DataFrame = None,
        specific_model_for_shaps: BaseEstimator = None,
        plot_shap: bool = True,
        plot_bar: bool = False
    ) -> None:
    """
    Generate the plots for the models evaluation.

    :param X: The data.
    :param X_train: The training data.
    :param X_test: The testing data.
    :param y_test: The testing labels.
    :param selector: The feature selector.
    :param features_array: The number of features to be selected.
    :param models_path: The path to the models.
    :param plots_path: The path to save the plots.
    :param shap_path: The path to save the shap plots.
    :param results_standard: The results for the standard models.
    :param results_tuned: The results for the tuned models.
    :param eval_metric: The evaluation metric.
    :param model_cv: The model used for cross-validation. If None, 
        the cross-validation plots will not be generated.
    :param results_cv: The results for the cross-validation. If None, 
        the cross-validation plots will not be generated.
    :param specific_model_for_shaps: A specific model to generate the shap plots. If None,
        only the plots for the best model will be generated. Default is None.
    :param plot_shap: Whether to plot the SHAP values. Default is True.
    :param plot_bar: Whether to plot the SHAP bar plot. Default is False.
    """
    plot_evals(plots_path, results_standard, results_tuned, eval_metric)
        
    if model_cv is not None and results_cv is not None:
        cv_model_name = get_model_name(model_cv, short=True)
        plot_boxplot(results_cv, cv_model_name, plots_path, eval_metric)

    bar_plot_path = shap_path / "bar_plot"

    if plot_shap:
        plot_shaps(
            X_train, X_test, X.columns,
            selector, features_array,
            models_path, shap_path,
        )
    if plot_bar:
        plot_shaps_bar_plot(
            X_train, X_test, y_test, X.columns, 
            selector, features_array,
            models_path, bar_plot_path
        )

    if specific_model_for_shaps is not None:
        specific_model_path = shap_path / get_model_name(specific_model_for_shaps, short=True).lower()
        specific_model_path_bar_plot = bar_plot_path / get_model_name(specific_model_for_shaps, short=True).lower()

        if plot_shap:
            plot_shaps(
                X_train, X_test, X.columns,
                selector, features_array,
                models_path, specific_model_path,
                specific_model=specific_model_for_shaps,
            )
        if plot_bar:
            plot_shaps_bar_plot(
                X_train, X_test, y_test, X.columns,
                selector, features_array,
                models_path, specific_model_path_bar_plot,
                specific_model=specific_model_for_shaps
            )


def move_pipeline_outputs(target_path: PathLike) -> None:
    """
    Move the pipeline outputs to a target directory.

    :param target_path: The target directory to move the outputs.
    """
    target_path = Path(target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    default_output = Path("output")
    for file in os.listdir(default_output):
        file = default_output / file
        shutil.move(file, target_path)
    
    print(f"Moved files to {target_path}")