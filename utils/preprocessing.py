from typing import List
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from utils.model_dumping import load_rfe_selector, save_model


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and preprocess it by dropping columns with more 
    than 10% of missing values.

    :param file_path: Path to the CSV file.
    :type file_path: str
    
    :return: The dataframe with the data.
    :rtype: pd.DataFrame
    """

    df = pd.read_csv(file_path)
    missing_percentage = df.isnull().mean() * 100
    df = df.drop(columns=missing_percentage[missing_percentage > 10].index)
    df = df.drop(columns=["patient_id"])
    return df


def preprocess_data(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Preprocess the data by imputing the most frequent value and scaling it.

    :param X_train: The training data.
    :type X_train: np.ndarray

    :param X_test: The testing data.
    :type X_test: np.ndarray

    :return: The preprocessed training and testing data.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    imputer = SimpleImputer(strategy="most_frequent")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def train_selectors(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> List[RFE]:
    """
    Train the Recursive Feature Elimination (RFE) selectors with different number of features.

    Load the selectors from the disk if they already exist. Otherwise, train them and save them.

    :param X_train: The training data.
    :type X_train: np.ndarray

    :param X_test: The testing data.
    :type X_test: np.ndarray

    :param y_train: The training labels.
    :type y_train: np.ndarray
    
    :param y_test: The testing labels.
    :type y_test: np.ndarray

    :return: An array with the trained selectors for different number of features.
    :rtype: List[RFE]
    """

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


def fit_selector(X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, selector: object) -> None:
    """
    Fit the selector in the whole dataset.

    :param X_train: The training data.
    :type X_train: np.ndarray

    :param X_test: The testing data.
    :type X_test: np.ndarray

    :param y_train: The training labels.
    :type y_train: np.ndarray

    :param y_test: The testing labels.
    :type y_test: np.ndarray

    :param selector: The selector to be fitted.
    :type selector: object

    :return: None
    :rtype: None
    """

    X = np.concatenate((X_train, X_test), axis=0)  # We decide do fit the selector in the whole dataset
    y = np.concatenate((y_train, y_test), axis=0)
    selector.fit(X, y)
