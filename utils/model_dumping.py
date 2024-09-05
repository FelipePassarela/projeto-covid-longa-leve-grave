import os
import pickle
from sklearn.feature_selection import RFE
from sklearn.svm import SVC


def load_rfe_selector(n_features):
    model_file = f"models/RFE_{n_features}feats.pkl"
    try:
        with open(model_file, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return RFE(SVC(kernel="linear", random_state=42), n_features_to_select=n_features)


def save_model(model, n_features):
    os.makedirs("models", exist_ok=True)
    model_file = f"models/{model.__class__.__name__}_{n_features}feats.pkl"
    with open(model_file, 'wb') as file:
        pickle.dump(model, file)
