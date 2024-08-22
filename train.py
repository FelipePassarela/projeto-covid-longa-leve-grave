import os
import pickle
import numpy as np
import pandas as pd
import warnings
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


def feature_selection(X_train, X_test, y_train, y_test, n_features):
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    selector = RFE(SVC(kernel="linear", random_state=42), n_features_to_select=n_features)
    selector = selector.fit(X, y)  # It has to be fitted with all data
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected


def save_model(model, n_features):
    os.makedirs("models", exist_ok=True)
    model_file = f"models/{model.__class__.__name__}_{n_features}feats_rfe.pkl"
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)


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

    models_and_params = (
        lr_model_and_params(),
        svm_model_and_params(),
        knn_model_and_params(),
        rf_model_and_params(),
        xgb_model_and_params()
    )

    for model, params in models_and_params:
        df_out = pd.DataFrame(columns=["n_features", "accuracy", "f1", "roc_auc", "confusion_matrix"])

        n_features_array = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        for n_features in n_features_array:
            print(f"\r{model.__class__.__name__} - {n_features} feature(s)", end="")

            X_train_selected, X_test_selected = feature_selection(X_train, X_test, y_train, y_test, n_features)

            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)

            new_row = pd.DataFrame({
                "n_features": [n_features],
                "accuracy": [accuracy_score(y_test, y_pred)],
                "f1": [f1_score(y_test, y_pred)],
                "roc_auc": [roc_auc_score(y_test, model.predict_proba(X_test_selected)[:, 1])],
                "confusion_matrix": [confusion_matrix(y_test, y_pred)]
            })
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                df_out = pd.concat([df_out, new_row], ignore_index=True)

            save_model(model, n_features)

        os.makedirs("results", exist_ok=True)
        df_out.to_csv(f"results/{model.__class__.__name__}_rfe.csv", index=False)

if __name__ == "__main__":
    main()
