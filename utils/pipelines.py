from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFE, SelectorMixin
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils.models.evaluate_models import (EvalResultsDict, evaluate_cv,
                                          evaluate_models)
from utils.models.models_and_params import get_model_and_params, get_model_name
from utils.plots.results import plot_boxplot, plot_evals
from utils.plots.shap import plot_shaps, plot_shaps_feature_clustering
from utils.preprocessing import fit_selector, load_data, preprocess_data


def main_pipeline(
        genomic_data_path: PathLike, 
        target: str,
        features_array: list[int],
        oversample: bool = False,
        fit_selector_on_whole_dataset: bool = False,
        missing_threshold: float = 10.0,
        run_cv: bool = True,
        specific_model_for_shaps: BaseEstimator = None
    ) -> None:
    """
    Executes the main pipeline of the project.

    Loads genomic data, preprocesses it, trains various machine learning models with different 
    feature sets, and evaluates their performance. Also generates visualization plots and
    SHAP value analysis.

    The workflow includes:
    1. Loading and preprocessing the data
    2. Splitting into train/test sets
    3. Feature selection using RFE
    4. Training and evaluating multiple models
    5. Generating performance plots and analysis visualizations

    :param genomic_data_path: Path to the genomic data file.
    :param target: The target variable for the dataset.
    :param features_array: List of number of features to be selected.
    :param oversample: Whether to oversample the data. Default is False.
    :param fit_selector_on_whole_dataset: Whether to fit the selector on the whole dataset. Default is False.
    :param missing_threshold: Maximum percentage of missing values allowed for a column to be kept. Default is 10.0.
    :param run_cv: Whether to run cross-validation. Default is True.
    :param specific_model_for_shaps: A specific model to generate SHAP plots for. If None,
        only the plots for the best model will be generated. Default is None.
    """
    df = load_data(genomic_data_path, target, missing_threshold=missing_threshold)
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test, oversample=oversample)

    selector_estim = SVC(kernel="linear", random_state=42)
    selector = RFE(selector_estim, n_features_to_select=1)
    fit_selector(
        X_train, X_test, y_train, y_test, 
        selector, on_whole_dataset=fit_selector_on_whole_dataset
    )

    models_and_params = [
        get_model_and_params("logistic_regression"),
        get_model_and_params("svm"),
        get_model_and_params("knn"),
        get_model_and_params("random_forest"),
        get_model_and_params("xgboost")
    ]

    models_path = Path("output/models/")
    results_path = Path("output/results/")
    plots_path = Path("output/plots/")
    shap_path = plots_path / "shap"
    eval_metric = "roc_auc"

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
        model_cv = SVC(random_state=42, probability=True)
        results_cv = evaluate_cv(
            np.concatenate([X_train, X_test]),
            np.concatenate([y_train, y_test]),
            model=model_cv,
            selector=selector,
            features_array=features_array,
            scoring=eval_metric,
            cv=5
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
        specific_model_for_shaps=specific_model_for_shaps
    )


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
    """
    plot_evals(plots_path, results_standard, results_tuned, eval_metric)
    
    plot_shaps(
        X_train, X_test, X.columns,
        selector, features_array,
        models_path, shap_path,
        comparison_metric=eval_metric
    )    
    plot_shaps_feature_clustering(
        X_train, X_test, y_test, X.columns, 
        selector, features_array,
        models_path, shap_path
    )

    if model_cv and results_cv is not None:
        cv_model_name = get_model_name(model_cv, short=True).lower()
        plot_boxplot(results_cv, cv_model_name, plots_path, eval_metric)

    if specific_model_for_shaps:
        specific_model_path = shap_path / get_model_name(specific_model_for_shaps, short=True).lower()
        plot_shaps(
            X_train, X_test, X.columns,
            selector, features_array,
            models_path, specific_model_path,
            specific_model=specific_model_for_shaps,
            comparison_metric=eval_metric
        )
        plot_shaps_feature_clustering(
            X_train, X_test, y_test, X.columns,
            selector, features_array,
            models_path, specific_model_path,
            specific_model=specific_model_for_shaps
        )


def move_pipeline_outputs(target_output_path: PathLike) -> None:
    """
    Move the pipeline outputs to a target directory.

    :param target_output_path: The path to the target directory.
    """
    target_output = Path(target_output_path)
    target_output.mkdir(parents=True, exist_ok=True)

    default_output = Path("output")
    for file in default_output.iterdir():
        if file.is_file():
            file.rename(target_output / file.name)
    
    print(f"Moved files to {target_output}")