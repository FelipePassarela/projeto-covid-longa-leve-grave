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
from utils.preprocessing import load_data, preprocess_data, train_selectors, oversample
from utils.plot_results import plot_all_results_subplots, plot_results, plot_shap, plot_umap_projection


FILE_NAME = "data/risk_1/no_vac.csv"
TARGET = "Risk"


def main() -> None:
    """
    Main execution function.
    
    Loads genomic data, preprocesses it, trains various machine learning models with different 
    feature sets, and evaluates their performance. Also generates visualization plots and
    SHAP value analysis.

    The workflow includes:
    1. Loading and preprocessing the data
    2. Splitting into train/test sets
    3. Oversampling to balance classes
    4. Feature selection using RFE
    5. Training and evaluating multiple models
    6. Generating performance plots and analysis visualizations
    """
    df = load_data(FILE_NAME)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test = preprocess_data(X_train, X_test)
    X_train, y_train = oversample(X_train, y_train)

    features_array = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    selector_array = train_selectors(X_train, X_test, y_train, y_test, features_array)

    models_and_params = [
        get_model_and_params("logistic_regression"),
        get_model_and_params("svm"),
        get_model_and_params("knn"),
        get_model_and_params("random_forest"),
        get_model_and_params("xgboost")
    ]

    evaluate_models(X_train, X_test, y_train, y_test, X.columns, selector_array, models_and_params, False)
    evaluate_models(X_train, X_test, y_train, y_test, X.columns, selector_array, models_and_params, True)

    evaluation_metric = "roc_auc"
    plot_results("train_standard", evaluation_metric)
    plot_results("test_standard", evaluation_metric)
    plot_results("train_tuned", evaluation_metric)
    plot_results("test_tuned", evaluation_metric)
    plot_all_results_subplots(evaluation_metric)
    plot_shap(X_train, X_test, X.columns, "models/", evaluation_metric)
    # plot_umap_projection(FILE_NAME)


if __name__ == "__main__":
    main()
