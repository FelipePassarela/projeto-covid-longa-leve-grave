import os
import warnings
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from utils.model_dumping import save_model


def evaluate_models(X_train, X_test, y_train, y_test, X_columns, selector_array, models_and_params, tune=False):
    for model, params in models_and_params:
        df_out_train = pd.DataFrame(columns=["n_features", "accuracy", "f1", "roc_auc", "confusion_matrix", "selected_features"])
        df_out_test = pd.DataFrame(columns=["n_features", "accuracy", "f1", "roc_auc", "confusion_matrix", "selected_features"])

        for selector in selector_array:
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

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                df_out_train = pd.concat([df_out_train, new_row_train], ignore_index=True)
                df_out_test = pd.concat([df_out_test, new_row_test], ignore_index=True)

            save_model(model, selector.n_features_)

        train_path = f"results/train/{"tuned" if tune else "standart"}"
        test_path = f"results/test/{"tuned" if tune else "standart"}"
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        df_out_train.to_csv(f"{train_path}/{model.__class__.__name__}.csv", index=False)
        df_out_test.to_csv(f"{test_path}/{model.__class__.__name__}.csv", index=False)
        print()