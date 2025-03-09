from os import PathLike

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

from utils.models.evaluate_models import EvalResultsDict
from utils.models.models_and_params import get_model_name
from utils.plots.shap import plot_shaps_feature_clustering
from utils.plots.results import (plot_boxplot, plot_evals)
from utils.plots.shap import plot_shaps


def plots_pipeline(
        X: pd.DataFrame | np.ndarray,
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        selector: SelectorMixin,
        features_array: list,
        models_path: PathLike,
        plots_path: PathLike,
        shap_path: PathLike,
        model_cv: BaseEstimator,
        results_standard: EvalResultsDict,
        results_tuned: EvalResultsDict,
        results_cv: pd.DataFrame,
        eval_metric: str,
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
    :param model_cv: The model used for cross-validation.
    :param results_standard: The results for the standard models.
    :param results_tuned: The results for the tuned models.
    :param results_cv: The results for the cross-validation.
    :param eval_metric: The evaluation metric.
    :param specific_model_for_shaps: A specific model to generate the shap plots. Default is None.
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
    
    cv_model_name = get_model_name(model_cv, short=True).lower()
    plot_boxplot(results_cv, cv_model_name, plots_path, eval_metric)