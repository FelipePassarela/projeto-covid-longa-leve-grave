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

import numpy as np
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils.pipelines import plots_pipeline
from utils.models.evaluate_models import evaluate_cv, evaluate_models
from utils.models.models_and_params import get_model_and_params, get_model_name
from utils.preprocessing import fit_selector, load_data, preprocess_data

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
    3. Feature selection using RFE
    4. Training and evaluating multiple models
    5. Generating performance plots and analysis visualizations
    """
    df = load_data(FILE_NAME, TARGET)   
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test, oversample=False)

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
    shap_path = plots_path / "shap"
    eval_metric = "roc_auc"

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

    model_cv = SVC(random_state=42, probability=True)
    results_cv = evaluate_cv(
        np.concatenate([X_train, X_test]),
        np.concatenate([y_train, y_test]),
        model=model_cv,
        selector=selector,
        features_array=features_array,
        scoring=eval_metric,
        cv=5
    )
    
    plots_pipeline(
        X, X_train, X_test, y_test,
        selector, features_array,
        models_path, plots_path, shap_path,
        results_standard, results_tuned, eval_metric,
        model_cv=model_cv, results_cv=results_cv,
        specific_model_for_shaps=SVC()
    )

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    def process_dataset(category,
        subcategory, dataset_name, target):
        global FILE_NAME, TARGET
        FILE_NAME = f"data/{category}/{subcategory}/{dataset_name}"
        TARGET = target
        main()
        
        output_dir = f"results/{category}/{subcategory}"
        os.makedirs(output_dir, exist_ok=True)
        for file in os.listdir("output"):
            os.rename(f"output/{file}", f"{output_dir}/{file}")

    datasets = [
        ("grave", "geral", "merged.csv", "risk"),
        ("grave", "nao_vacinados", "merged.csv", "risk"),

        ("longa", "geral", "merged.csv", "Long_COVID"),
        ("longa", "nao_vacinados", "merged.csv", "Long_COVID"),

        ("dor", "nao_vacinados", "merged.csv", "Pain_Block"),
        ("sistema_cardiovascular", "nao_vacinados", "merged.csv", "Cardiovascular_sequelae"),
        ("sistema_nervoso", "nao_vacinados", "merged.csv", "Sist_Nerv_Per_"),
    ]

    for category, subcategory, file_name, target in datasets:
        print(f"Processing dataset: {category}/{subcategory}/{file_name} with target: {target}")
        process_dataset(category, subcategory, file_name, target)
