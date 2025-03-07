import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from utils.models_and_params import get_model_name
from utils.plot_results import SCORE_TITLES
from utils.preprocessing import load_data, oversample, preprocess_data


def evaluate_cv(X, y, model: BaseEstimator, features_array: list[int], scoring="roc_auc", cv=5) -> pd.DataFrame:
    results = []
    for n_features in features_array:
        print(f"Training with {n_features} features...")

        scores = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Fold {fold + 1}/{cv}")

            X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[val_idx]
            
            X_train, X_test = preprocess_data(X_train, X_test)
            X_train, y_train = oversample(X_train, y_train)

            model = model.__class__(**model.get_params())  # Reset model to initial state
            
            # We decide to fit the selector on the whole dataset
            X_full = np.concatenate((X_train, X_test), axis=0)
            y_full = np.concatenate((y_train, y_test), axis=0)

            selector = RFE(model, n_features_to_select=n_features)
            selector.fit(X_full, y_full)
            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)
            
            model.fit(X_train_selected, y_train)
            score = roc_auc_score(y_test, model.predict_proba(X_test_selected)[:, 1])
            scores.append(score)

        results.append({
            "n_features": n_features,
            "scores": np.array(scores),
            "mean_score": np.mean(scores),
            "std_score": np.std(scores)
        })
        print(f"Features: {n_features}, Mean {scoring}: {results[-1]['mean_score']:.4f} ± {results[-1]['std_score']:.4f}")
    
    return pd.DataFrame(results)


def plot_boxplot(results_df: pd.DataFrame, model_name, scoring="roc_auc", cv=5) -> plt.Figure:
    scores_per_feature = {n: scores for n, scores in zip(results_df["n_features"], results_df["scores"])}
    score_title = SCORE_TITLES.get(scoring, scoring)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(list(scores_per_feature.values()), tick_labels=list(scores_per_feature.keys()))
    ax.set_xlabel("Number of SNPs", fontsize=12)
    ax.set_ylabel(score_title, fontsize=12)
    ax.set_title(f"Distribution of {score_title} for {model_name} ({cv}-fold CV)", fontsize=14)
    plt.tight_layout()

    return fig


def process_dataset(category, subcategory, dataset_name, target):
    dataset_path = Path(f"data/28_01/{category}/{subcategory}/{dataset_name}")
    print(f"Processing {dataset_path}...")

    df = load_data(dataset_path)
    X = df.drop(columns=[target])
    y = df[target]

    model = LogisticRegression(max_iter=1000, random_state=42)
    features_array = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    scoring = "roc_auc"
    cv = 5
    results_df = evaluate_cv(X, y, model, features_array, scoring=scoring, cv=cv)
    
    output_folder = Path(f"output/{category}/{subcategory}")
    os.makedirs(output_folder, exist_ok=True)
    
    results_path = output_folder / "results.csv"
    results_df.to_csv(results_path, index=False)
    
    model_name = get_model_name(model)
    fig = plot_boxplot(results_df, model_name, scoring=scoring, cv=cv)
    plot_path = output_folder / "plots" / "boxplot.png"
    os.makedirs(plot_path.parent, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)


if __name__ == "__main__":
    matplotlib.use("Agg")

    datasets = [
        ("risk", "geral", "MATRIZ_GERAL_FILTRADO.csv", "Risk"),
        ("risk", "nao_vacinados_duas_doses", "matriz_genotipos_no_vac_COVID_GERAL_filtrado.csv", "Risk"),
        
        ("longa", "geral", "MATRIZ_GERAL_FILTRADO.csv", "Long Covid"),
        ("longa", "nao_vacinados_duas_doses", "matriz_genotipos_no_vac_COVID_LONGA_DUAS_DOSES_filtrado.csv", "Long Covid"),
    ]

    for category, subcategory, file_name, target in datasets:
        process_dataset(category, subcategory, file_name, target)