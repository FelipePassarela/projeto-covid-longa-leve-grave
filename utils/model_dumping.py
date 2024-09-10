import os
import pickle
from sklearn.feature_selection import RFE
from sklearn.svm import SVC


def load_rfe_selector(n_features: int) -> RFE:
    """
    Load the Recursive Feature Elimination (RFE) selector with a specific number of features.

    If the selector does not exist in the disk, create a new one.

    :param n_features: The number of features to be selected.
    :type n_features: int

    :return: The RFE selector.
    :rtype: RFE
    """

    model_file = f"models/RFE_{n_features}feats.pkl"
    try:
        with open(model_file, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return RFE(SVC(kernel="linear", random_state=42), n_features_to_select=n_features)


def save_model(model: object, n_features: int) -> None:
    """
    Save the model in the disk.

    :param model: The model to be saved.
    :type model: object

    :param n_features: The number of features used to train the model.
    :type n_features: int

    :return: None
    :rtype: None
    """

    os.makedirs("models", exist_ok=True)
    model_file = f"models/{model.__class__.__name__}_{n_features}feats.pkl"
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)
