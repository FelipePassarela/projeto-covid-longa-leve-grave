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

import os
from pathlib import Path

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils.evaluate_models import evaluate_models
from utils.models_and_params import get_model_and_params
from utils.plot_results import plot_evals, plot_shap
from utils.preprocessing import (fit_selector, load_data, oversample, preprocess_data,
                                 train_selectors)

FILE_NAME = "data/28_01/longa/nao_vacinados_uma_dose/matriz_genotipos_no_vac_COVID_LONGA_UMA__DOSE_filtrado.csv"
TARGET = "Long Covid"


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
    # X_train, y_train = oversample(X_train, y_train)

    selector_estim = SVC(kernel="linear", random_state=42)
    selector = RFE(selector_estim, n_features_to_select=1)
    features_array = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    fit_selector(X_train, X_test, y_train, y_test, selector, on_whole_dataset=True)

    models_and_params = [
        get_model_and_params("logistic_regression"),
        get_model_and_params("svm"),
        get_model_and_params("knn"),
        get_model_and_params("random_forest"),
        get_model_and_params("xgboost")
    ]

    models_path = Path("output/models/")
    results_path = Path("output/results/")
    plots_path = Path("output/plots/")

    results_standard = evaluate_models(
        X_train, X_test, y_train, y_test, X.columns, 
        selector, features_array, models_and_params,
        models_path, results_path, tune=False
    )
    results_tuned = evaluate_models(
        X_train, X_test, y_train, y_test, X.columns, 
        selector, features_array, models_and_params,
        models_path, results_path, tune=True
    )

    eval_metric = "roc_auc"
    plot_evals(plots_path, results_standard, results_tuned, eval_metric)
    plot_shap(X_train, X_test, X.columns, selector, features_array, models_path, eval_metric)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    def process_dataset(category, subcategory, dataset_name, target):
        global FILE_NAME, TARGET
        FILE_NAME = f"data/{category}/{subcategory}/{dataset_name}"
        TARGET = target
        main()
        
        output_dir = f"{category}/{subcategory}"
        os.makedirs(output_dir, exist_ok=True)
        for file in os.listdir("output"):
            os.rename(f"output/{file}", f"{output_dir}/{file}")

    # TODO: Make the boxplot
    # TODO: Make shap plot of SVM

    datasets = [
        ("dani", "geral", "MATRIZ_GERAL_FILTRADO_merged.csv", "Cardiovascular sequelae"),
        # ("dani", "nao_vacinados", "matriz_genotipos_no_vac_COVID_GERAL_filtrado_merged.csv", "Cardiovascular sequelae"),

        # ("mion", "geral", "MATRIZ_GERAL_FILTRADO_merged.csv", "Pain Block_175"),
        # ("mion", "nao_vacinados", "matriz_genotipos_no_vac_COVID_GERAL_filtrado_merged.csv", "Pain Block_175"),
    ]

    for category, subcategory, file_name, target in datasets:
        process_dataset(category, subcategory, file_name, target)
