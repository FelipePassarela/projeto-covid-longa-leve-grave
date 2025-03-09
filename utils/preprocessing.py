from typing import List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import RFE, SelectorMixin
from sklearn.impute import KNNImputer, SimpleImputer

from utils.models.model_dumping import load_rfe_selector, save_model


def load_data(file_path: str, target: str, threshold: float = 10.0) -> pd.DataFrame:
    """
    Load data from a CSV file and preprocess it by dropping columns with more 
    than a specified percentage of missing values.

    :param file_path: Path to the CSV file.
    :param target: The target variable for the dataset.
    :param threshold: Maximum percentage of missing values allowed for a column to be kept.

    :return: The dataframe with the data.
    """
    df = pd.read_csv(file_path)
    df = df.dropna(subset=[target])
    missing_percentage = df.isnull().mean() * 100
    df = df.drop(columns=missing_percentage[missing_percentage > threshold].index)
    df = df.drop(columns=["id"])
    return df


def preprocess_data(
        X_train: pd.DataFrame | np.ndarray, 
        X_test: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        y_test: pd.Series | np.ndarray,
        oversample: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the data by imputing the most frequent value and scaling it.

    :param X_train: The training data.
    :param X_test: The testing data.
    :param y_train: The training labels.
    :param y_test: The testing labels.
    :param oversample: Whether to oversample the minority class or not.

    :return: X_train, X_test, y_train, y_test
    """
    imputer = SimpleImputer(strategy="most_frequent")
    # imputer = KNNImputer(n_neighbors=5)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if oversample:
        X_train, y_train = oversample(X_train, y_train)

    return X_train, X_test, y_train, y_test


def train_selectors(
        X_train: pd.DataFrame | np.ndarray, 
        X_test: pd.DataFrame | np.ndarray, 
        y_train: pd.Series | np.ndarray, 
        y_test: pd.Series | np.ndarray, 
        features_array: List[int],
        on_whole_dataset: bool = False
    ) -> List[RFE]:
    """
    Train the Recursive Feature Elimination (RFE) selectors with different number of features.

    Load the selectors from the disk if they already exist. Otherwise, train them and save them.

    :param X_train: The training data.
    :param X_test: The testing data.
    :param y_train: The training labels.
    :param y_test: The testing labels.
    :param features_array: The number of features to be selected by each selector.
    :param on_whole_dataset: Whether to fit the selector in the whole dataset.

    :return: An array with the trained selectors for different number of features.
    """
    selector_array = [load_rfe_selector(n_features, "output/models/selectors") for n_features in features_array]

    for selector in selector_array:
        if hasattr(selector, "n_features_"):  # If it was already trained,
            continue
        fit_selector(X_train, X_test, y_train, y_test, selector, on_whole_dataset=on_whole_dataset)
        save_model(selector, selector.n_features_to_select, "output/models/selectors")
    print()

    return selector_array


def fit_selector(
        X_train: pd.DataFrame | np.ndarray, 
        X_test: pd.DataFrame | np.ndarray, 
        y_train: pd.Series | np.ndarray, 
        y_test: pd.Series | np.ndarray, 
        selector: SelectorMixin,
        on_whole_dataset: bool = False
    ) -> None:
    """
    Fit the selector in the whole dataset.

    :param X_train: The training data.
    :param X_test: The testing data.
    :param y_train: The training labels.
    :param y_test: The testing labels.
    :param selector: The selector to be fitted.
    :param on_whole_dataset: Whether to fit the selector in the whole dataset.
    """
    print(f"Fitting {selector.__class__.__name__} with {selector.n_features_to_select} feature(s)...")
    if on_whole_dataset:
        # We decided to fit the selector on the whole dataset.
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        selector.fit(X, y)
    else:
        selector.fit(X_train, y_train)


def oversample(
        X_train: pd.DataFrame | np.ndarray, 
        y_train: pd.DataFrame | np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oversample the minority class in the training data.

    :param X_train: The training data.
    :param y_train: The training labels.
    
    :return: The oversampled training data and labels.
    """
    adasyn = ADASYN(sampling_strategy='minority', random_state=42)
    X_train_resamp, y_train_resamp = adasyn.fit_resample(X_train, y_train)
    return X_train_resamp, y_train_resamp
