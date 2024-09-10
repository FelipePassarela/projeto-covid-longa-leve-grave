import os
from typing import List, Tuple
import warnings
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from utils.model_dumping import save_model
from utils.models_and_params import HyperParamGrid


def evaluate_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, 
                    y_test: pd.Series, X_columns: List[str], selector_array: List[BaseEstimator],
                    models_and_params: List[Tuple[BaseEstimator, HyperParamGrid]], tune: bool) -> None:
    """
    Evaluate the models with the given data and hyperparameters and save the 
    results in CSV files at the results folder.

    :param X_train: The training data.
    :type X_train: pd.DataFrame

    :param X_test: The testing data.
    :type X_test: pd.DataFrame

    :param y_train: The training labels.
    :type y_train: pd.Series

    :param y_test: The testing labels.
    :type y_test: pd.Series

    :param X_columns: The columns of the data.
    :type X_columns: List[str]

    :param selector_array: The feature selectors for each number of features.
    :type selector_array: List[BaseEstimator]

    :param models_and_params: The models and hyperparameters.
    :type models_and_params: List[Tuple[BaseEstimator, HyperParamGrid]]

    :param tune: Whether to tune the hyperparameters or not.
    :type tune: bool

    :return: None
    :rtype: None
    """

    for model, params in models_and_params:
        df_out_train = pd.DataFrame(columns=["n_features", "accuracy", "f1", "roc_auc", "confusion_matrix", "selected_features"])
        df_out_test = pd.DataFrame(columns=["n_features", "accuracy", "f1", "roc_auc", "confusion_matrix", "selected_features"])

        for selector in selector_array:
            if tune:
                print(f"\rTuning {model.__class__.__name__ } with {selector.n_features_to_select} feature(s)...", end="")
            else:
                print(f"\rTraining {model.__class__.__name__} with {selector.n_features_to_select} feature(s)...", end="")

            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)

            if tune:
                grid_search = GridSearchCV(model, params, cv=5, scoring="roc_auc", n_jobs=-1)
                grid_search.fit(X_train_selected, y_train)
                model = grid_search.best_estimator_
            else:
                model.fit(X_train_selected, y_train)

            y_pred_train = model.predict(X_train_selected)
            y_pred_test = model.predict(X_test_selected)

            new_row_train = pd.DataFrame({
                "n_features"       : [selector.n_features_],
                "accuracy"         : [accuracy_score(y_train, y_pred_train)],
                "f1"               : [f1_score(y_train, y_pred_train)],
                "roc_auc"          : [roc_auc_score(y_train, model.predict_proba(X_train_selected)[:, 1])],
                "confusion_matrix" : [confusion_matrix(y_train, y_pred_train)],
                "selected_features": [X_columns[selector.support_]]
            })
            new_row_test = pd.DataFrame({
                "n_features"       : [selector.n_features_],
                "accuracy"         : [accuracy_score(y_test, y_pred_test)],
                "f1"               : [f1_score(y_test, y_pred_test)],
                "roc_auc"          : [roc_auc_score(y_test, model.predict_proba(X_test_selected)[:, 1])],
                "confusion_matrix" : [confusion_matrix(y_test, y_pred_test)],
                "selected_features": [X_columns[selector.support_]]
            })

            if tune:
                new_row_train["parameters"] = [model.get_params()]
                new_row_test["parameters"] = [model.get_params()]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                df_out_train = pd.concat([df_out_train, new_row_train], ignore_index=True)
                df_out_test = pd.concat([df_out_test, new_row_test], ignore_index=True)

            save_model(model, selector.n_features_)

        train_path = f"results/train/{"tuned" if tune else "standard"}"
        test_path = f"results/test/{"tuned" if tune else "standard"}"
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        df_out_train.to_csv(f"{train_path}/{model.__class__.__name__}.csv", index=False)
        df_out_test.to_csv(f"{test_path}/{model.__class__.__name__}.csv", index=False)
        print()