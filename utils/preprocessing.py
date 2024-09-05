import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from utils.model_dumping import load_rfe_selector, save_model


def load_data(file_path):
    df = pd.read_csv(file_path)
    missing_percentage = df.isnull().mean() * 100
    df = df.drop(columns=missing_percentage[missing_percentage > 10].index)
    df = df.drop(columns=["patient_id"])
    return df


def preprocess_data(X_train, X_test):
    imputer = SimpleImputer(strategy="most_frequent")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def train_selectors(X_train, X_test, y_train, y_test):
    n_features_array = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    selector_array = [
        load_rfe_selector(n_features) for n_features in n_features_array
    ]

    for selector in selector_array:
        if hasattr(selector, "n_features_"):  # If it was already trained,
            continue

        print(f"\rTraining {selector.__class__.__name__} with {selector.n_features_to_select} feature(s)...", end="")
        fit_selector(X_train, X_test, y_train, y_test, selector)
        save_model(selector, selector.n_features_to_select)
    print()

    return selector_array


def fit_selector(X_train, X_test, y_train, y_test, selector):
    X = np.concatenate((X_train, X_test), axis=0)  # We decide do fit the selector in the whole dataset
    y = np.concatenate((y_train, y_test), axis=0)
    selector.fit(X, y)
