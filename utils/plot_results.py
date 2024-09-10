import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.preprocessing import load_data, preprocess_data
from umap import UMAP


SCORE_TITLES = {
    "accuracy": "Accuracy",
    "f1"      : "F1 Score",
    "roc_auc" : "ROC AUC",
}

RESULTS_PATHS = {
    'test_standard' : 'results/test/standard',
    'train_standard': 'results/train/standard',
    'test_tuned'    : 'results/test/tuned',
    'train_tuned'   : 'results/train/tuned',
}

SUBPLOT_TITLES = {
    'test_standard' : 'Test (Standard)',
    'train_standard': 'Train (Standard)',
    'test_tuned'    : 'Test (Tuned)',
    'train_tuned'   : 'Train (Tuned)',
}


def plot_results(results_type: str, score: str) -> None:
    """
    Plots a figure comparing different models' scores for different number of features.

    :param results_type: Type of the results to be plotted. Should be one of the keys in RESULTS_PATHS.
    :type path: str

    :param score: Score to be plotted. Should be one of the score columns in the CSV files.
    :type score: str

    :return: None
    :rtype: None
    """

    plt.figure(figsize=(10, 6))

    path = RESULTS_PATHS[results_type]
    for file in os.listdir(path):
        if file.endswith(".csv"):
            model_name = file.split(".")[0]
            data = pd.read_csv(os.path.join(path, file))
            plt.plot(data["n_features"], data[score], label=model_name, marker='o')

    title = f"{SCORE_TITLES[score]} of Different Models - {SUBPLOT_TITLES[results_type]}"

    plt.title(title)
    plt.xticks(data["n_features"])
    plt.xlabel("Number of SNPs", fontsize=12)
    plt.ylabel(SCORE_TITLES[score], fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{title.replace(" ", "_").lower()}.png")
    plt.close()


def plot_all_results_subplots(score: str) -> None:
    """
    Plots a figure with multiple subplots comparing different models' scores for different number of features.

    The data should already be saved and split in the following directories:
    - results/test/standard
    - results/train/standard
    - results/test/tuned
    - results/train/tuned

    :param score: Score to be plotted. Should be one of the score columns in the CSV files.
    :type score: str

    :return: None
    :rtype: None
    """

    # Reading all CSV files and concatenating them
    data_list = []
    for key, value in RESULTS_PATHS.items():
        for file in os.listdir(value):
            if file.endswith(".csv"):
                data = pd.read_csv(os.path.join(value, file))
                data["model"] = file.split(".")[0]
                data["type"] = key
                data_list.append(data)

    data = pd.concat(data_list, ignore_index=True)
    data["n_features"] = data["n_features"].astype(str)
    
    main_title = "Standard vs Tuned Comparison (Train and Test)"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    fig.suptitle(main_title, fontsize=18)

    # Plotting subplots
    for i, (key, value) in enumerate(RESULTS_PATHS.items()):
        ax = axes[i // 2, i % 2]
        
        for model in data["model"].unique():
            model_data = data[(data["model"] == model) & (data["type"] == key)]
            ax.plot(model_data["n_features"], model_data[score], label=model, marker='o')
        ax.set_title(SUBPLOT_TITLES[key])
        ax.set_ylim(0.5, 1.025)
        ax.grid(True, which='both', linestyle='--')
        ax.legend()
        
        if i == 0 or i == 1:
            ax.tick_params(axis='x', which='both', length=0)
        if i == 1 or i == 3:
            ax.tick_params(axis='y', which='both', length=0) 
    
    fig.text(0.5, 0.02, "Number of SNPs", ha='center', va='center', fontsize=14)
    fig.text(0.03, 0.5, SCORE_TITLES[score], ha='center', va='center', rotation='vertical', fontsize=14)
    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{main_title.replace(' ', '_').lower()}.png")
    plt.close()


def plot_umap_projection(data_path: str) -> None:
    """
    Plot the UMAP projection of the data with the selected features.

    Load the selectors from the disk and perform same transformations as in the evaluate_models.py script. Then, reduce
    the dimensionality of the data with UMAP and save the plot at the plots/umap folder.

    :param data_path: Path to the CSV file with the data.
    :type data_path: str
    
    :return: None
    :rtype: None
    """

    df = load_data(data_path)

    for i in [3, *range(5, 51, 5)]:
        with open(f"models/RFE_{i}feats.pkl", "rb") as f:
            selector = pickle.load(f)

        X = df.drop(columns=["risk"])
        y = df["risk"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_test = preprocess_data(X_train, X_test)

        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        reducer = UMAP(random_state=42)
        X_embedded = reducer.fit_transform(X)

        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis')
        plt.title(f"UMAP projection of the data\n({selector.n_features_} selected features)")
        plt.gca().set_aspect('equal', 'datalim')
        plt.grid(False)

        handle, _ = scatter.legend_elements()
        plt.legend(handle, ["Low Risk", "High Risk"], title="Risk")
        plt.tight_layout()

        os.makedirs("plots/umap", exist_ok=True)
        plt.savefig(f"plots/umap/UMAP_{selector.n_features_}feats.png")
        plt.clf()
        plt.close()
