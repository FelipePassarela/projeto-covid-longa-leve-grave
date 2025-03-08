import warnings
from os import PathLike
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFE
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC

from utils.model_dumping import save_model
from utils.models_and_params import HyperParamGrid, get_model_name

EvalResultsDict = Dict[str, Dict[str, pd.DataFrame]]


def evaluate_models(
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        y_test: pd.Series | np.ndarray,
        X_columns: pd.Index,
        selector: RFE,
        feature_array: Sequence[int],
        models_and_params: Sequence[Tuple[BaseEstimator, HyperParamGrid]],
        model_path: PathLike,
        results_path: PathLike,
        tune: bool = False
    ) -> EvalResultsDict:
    """
    Evaluate the models with the given data and hyperparameters.

    :param X_train: The training data.
    :param X_test: The testing data.
    :param y_train: The training labels.
    :param y_test: The testing labels.
    :param X_columns: The columns of the data.
    :param selector: The fitted feature selector.
    :param feature_array: The feature indexes for each number of features to be selected.
    :param models_and_params: The models and hyperparameters.
    :param model_path: The path to save the models.
    :param results_path: The path to save the results.
    :param tune: Whether to tune the hyperparameters or not.

    :return: A dictionary with the results for each model.
    """
    results = {}

    for model, params in models_and_params:
        df_out_train = pd.DataFrame()
        df_out_test = pd.DataFrame()
        model_name = get_model_name(model, short=True)

        for n_feat in feature_array:
            if tune:
                print(f"\rTuning {model_name} with {n_feat} feature(s)...", end="")
            else:
                print(f"\rTraining {model_name} with {n_feat} feature(s)...", end="")

            X_train_selected, _ = extract_subset(selector, X_train, n_feat)
            X_test_selected, feat_indices = extract_subset(selector, X_test, n_feat)

            if tune:
                grid_search = GridSearchCV(model, params, cv=5, scoring="roc_auc", n_jobs=-1)
                grid_search.fit(X_train_selected, y_train)
                model = grid_search.best_estimator_
            else:
                model.fit(X_train_selected, y_train)

            y_pred_train = model.predict(X_train_selected)
            y_pred_test = model.predict(X_test_selected)

            new_row_train = pd.DataFrame({
                "n_features"       : [n_feat],
                "accuracy"         : [accuracy_score(y_train, y_pred_train)],
                "f1"               : [f1_score(y_train, y_pred_train)],
                "roc_auc"          : [roc_auc_score(y_train, model.predict_proba(X_train_selected)[:, 1])],
                "confusion_matrix" : [confusion_matrix(y_train, y_pred_train)],
                "selected_features": [X_columns[feat_indices]]
            })
            new_row_test = pd.DataFrame({
                "n_features"       : [n_feat],
                "accuracy"         : [accuracy_score(y_test, y_pred_test)],
                "f1"               : [f1_score(y_test, y_pred_test)],
                "roc_auc"          : [roc_auc_score(y_test, model.predict_proba(X_test_selected)[:, 1])],
                "confusion_matrix" : [confusion_matrix(y_test, y_pred_test)],
                "selected_features": [X_columns[feat_indices]]
            })

            if tune:
                new_row_train["parameters"] = [model.get_params()]
                new_row_test["parameters"] = [model.get_params()]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                df_out_train = pd.concat([df_out_train, new_row_train], ignore_index=True)
                df_out_test = pd.concat([df_out_test, new_row_test], ignore_index=True)

            path = Path(f"{model_path}/{'tuned' if tune else 'standard'}")
            save_model(model, n_feat, path)

        results[model_name] = {"train": df_out_train, "test": df_out_test}
        print()
    
    save_model_results(results, results_path, tuned=tune)
    return results


def evaluate_cv(
        X: pd.DataFrame | np.ndarray, 
        y: pd.Series | np.ndarray, 
        model: BaseEstimator, 
        selector: RFE,
        features_array: Sequence[int], 
        scoring: str = "roc_auc",
        cv: int = 5
    ) -> pd.DataFrame:
    """
    Evaluate the model with cross-validation.

    :param X: The data.
    :param y: The labels.
    :param model: The machine learning model to evaluate.
    :param selector: The feature selector.
    :param features_array: The number of features to evaluate.
    :param scoring: The scoring metric.
    :param cv: The number of cross-validation folds.
        
    :return: A DataFrame containing the cross-validation results.
    """
    results = []
    for n_feats in features_array:
        scores = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Fold {fold + 1}/{cv}")

            X_train, X_test = X[train_idx], X[val_idx]
            y_train, y_test = y[train_idx], y[val_idx]

            X_train_selected, _ = extract_subset(selector, X_train, n_feats)
            X_test_selected, _ = extract_subset(selector, X_test, n_feats)

            model = model.__class__(**model.get_params())  # Reset model to initial state

            model.fit(X_train_selected, y_train)
            scoring_fn = get_scoring_fn(scoring)
            if scoring == "roc_auc":
                score = scoring_fn(y_test, model.predict_proba(X_test_selected)[:, 1])
            else:
                score = scoring_fn(y_test, model.predict(X_test_selected))
            scores.append(score)

        results.append({'n_features': n_feats, 'scores': scores})

    return pd.DataFrame(results)


def save_model_results(results: pd.DataFrame, path: PathLike, tuned: bool = False) -> None:
    """
    Save the results of the models in the disk.

    :param results: The results of the models.
    :param path: The path to save the results.
    :param tuned: Whether the models are tuned or not.
    """
    eval_mode = "tuned" if tuned else "standard"
    train_path = Path(path) / "train" / eval_mode
    test_path = Path(path) / "test" / eval_mode
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    for model_name, res in results.items():
        train_df = res["train"]
        test_df = res["test"]
        train_df.to_csv(train_path / f"{model_name}.csv", index=False)
        test_df.to_csv(test_path / f"{model_name}.csv", index=False)


def extract_subset(selector, X, n_feats: int):
    """
    Extract the subset of features selected by the selector.

    :param selector: The selector.
    :param X: The data.
    :param n_feats: The number of features to select.

    :return: The subset of features selected by the selector and
                the indices of the selected features.
    """
    feature_indices = np.argsort(selector.ranking_)[:n_feats]
    subset = X[:, feature_indices]
    return subset, feature_indices


def get_scoring_fn(scoring: str):
    """
    Get the scoring function based on the metric.

    :param scoring: The metric to evaluate.

    :return: The scoring function.
    """
    if scoring == "accuracy":
        return accuracy_score
    elif scoring == "f1":
        return f1_score
    elif scoring == "roc_auc":
        return roc_auc_score
    else:
        raise ValueError(f"Unknown scoring metric: {scoring}")
