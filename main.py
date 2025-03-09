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


from pathlib import Path

import matplotlib
from sklearn.svm import SVC

from utils.pipelines import main_pipeline, move_pipeline_outputs


def main():
    matplotlib.use("Agg")

    datasets = [
        ("grave", "geral", "merged.csv", "risk"),
        ("grave", "nao_vacinados", "merged.csv", "risk"),

        ("longa", "geral", "merged.csv", "Long_COVID"),
        ("longa", "nao_vacinados", "merged.csv", "Long_COVID"),

        ("dor", "nao_vacinados", "merged.csv", "Pain_Block"),
        ("sistema_cardiovascular", "nao_vacinados", "merged.csv", "Cardiovascular_sequelae"),
        ("sistema_nervoso", "nao_vacinados", "merged.csv", "Sist_Nerv_Per_"),
    ]

    FEATURES_ARRAY = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    for category, subcategory, file_name, target in datasets:
        print(f"Processing dataset: {category}/{subcategory}/{file_name} with target: {target}")

        genomic_data_path = Path(f"data/{category}/{subcategory}/{file_name}")
        
        main_pipeline(
            genomic_data_path,
            target,
            FEATURES_ARRAY,
            oversample=False,
            fit_selector_on_whole_dataset=True,
            missing_threshold=10.0,
            run_cv=True,
            specific_model_for_shaps=SVC(),
            eval_metric="roc_auc",
            plot_shap=True,
            plot_bar=False
        )

        target_output = Path(f"results/{category}/{subcategory}")
        move_pipeline_outputs(target_output)


if __name__ == "__main__":
    main()
