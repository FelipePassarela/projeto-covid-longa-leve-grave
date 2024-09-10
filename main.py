#!/usr/bin/env python3

"""Main script of the project. 

This script loads genomic data, preprocesses it, trains multiple machine learning models, 
and evaluates their performance. It also generates plots to visualize the model results 
and projects the data into lower dimensions using UMAP.

Usage:
    python main.py
"""

__author__ = "Felipe dos Santos Passarela"
__email__ = "felipepassarela11@gmail.com"

from sklearn.model_selection import train_test_split
from utils.evaluate_models import evaluate_models
from utils.models_and_params import get_model_and_params
from utils.preprocessing import load_data, preprocess_data, train_selectors
from utils.plot_results import plot_all_results_subplots, plot_results, plot_umap_projection


FILE_NAME = "final_genotipos_GERAL_RISK.csv"


def main():
    df = load_data(FILE_NAME)

    X = df.drop(columns=["risk"])
    y = df["risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train, X_test = preprocess_data(X_train, X_test)
    selector_array = train_selectors(X_train, X_test, y_train, y_test)

    models_and_params = [
        get_model_and_params("logistic_regression"),
        get_model_and_params("svm"),
        get_model_and_params("knn"),
        get_model_and_params("random_forest"),
        get_model_and_params("xgboost")
    ]

    evaluate_models(X_train, X_test, y_train, y_test, X.columns, selector_array, models_and_params, False)
    evaluate_models(X_train, X_test, y_train, y_test, X.columns, selector_array, models_and_params, True)

    plot_results("train_standard", "roc_auc")
    plot_results("test_standard", "roc_auc")
    plot_results("train_tuned", "roc_auc")
    plot_results("test_tuned", "roc_auc")
    plot_all_results_subplots("roc_auc")
    plot_umap_projection(FILE_NAME)


if __name__ == "__main__":
    main()
