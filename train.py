import os
import pickle
import numpy as np
import pandas as pd
import warnings
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def svm_model_and_params():
    return SVC(random_state=42, probability=True), {  # probability=True may be incosistent with predict method
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


def fit_selector(X_train, X_test, y_train, y_test, selector):
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    selector.fit(X, y)  # It has to be fitted with all data


def save_model(model, n_features):
    os.makedirs("models", exist_ok=True)
    model_file = f"models/{model.__class__.__name__}_{n_features}feats_rfe.pkl"
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)


def load_model(model_name, n_features):
    model_file = f"models/{model_name}_{n_features}feats_rfe.pkl"
    try:
        with open(model_file, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return RFE(SVC(kernel="linear", random_state=42), n_features_to_select=n_features)


def main():
    df = pd.read_csv("final_genotipos_GERAL_RISK.csv")
    missing_percentage = df.isnull().mean() * 100
    df = df.drop(columns=missing_percentage[missing_percentage > 10].index)
    df = df.drop(columns=["patient_id"])

    X = df.drop(columns=["risk"])
    y = df["risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    imputer = SimpleImputer(strategy="most_frequent")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_features_array = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    selector_array = [
        load_model("RFE", n_features) for n_features in n_features_array
    ]
    models_and_params = (
        lr_model_and_params(),
        svm_model_and_params(),
        knn_model_and_params(),
        rf_model_and_params(),
        xgb_model_and_params()
    )

    for selector in selector_array:
        if not hasattr(selector, "n_features_"):  # If it was not fitted
            print(f"\rTraining RFE with {selector.n_features_to_select} feature(s)...", end="")
            fit_selector(X_train, X_test, y_train, y_test, selector)
            save_model(selector, selector.n_features_to_select)
    print()

    for model, params in models_and_params:
        df_out = pd.DataFrame(columns=["n_features", "accuracy", "f1", "roc_auc", "confusion_matrix"])

        for selector in selector_array:
            print(f"\rTraining {model.__class__.__name__} with {selector.n_features_to_select} feature(s)...", end="") 

            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)
            
            # grid_search = GridSearchCV(model, params, cv=5, scoring="auc_roc", n_jobs=-1)
            # grid_search.fit(X_train_selected, y_train)
            # model = grid_search.best_estimator_
            # y_pred = model.predict(X_test_selected)

            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)

            new_row = pd.DataFrame({
                "n_features": [selector.n_features_],
                "accuracy": [accuracy_score(y_test, y_pred)],
                "f1": [f1_score(y_test, y_pred)],
                "roc_auc": [roc_auc_score(y_test, model.predict_proba(X_test_selected)[:, 1])],
                "confusion_matrix": [confusion_matrix(y_test, y_pred)],
                "selected_features": [X.columns[selector.support_]]
            })
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                df_out = pd.concat([df_out, new_row], ignore_index=True)
            
            save_model(model, selector.n_features_)

        os.makedirs("results", exist_ok=True)
        df_out.to_csv(f"results/{model.__class__.__name__}_rfe.csv", index=False)
        print()

if __name__ == "__main__":
    main()
