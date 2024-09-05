import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_results(path, score, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))

    for file in os.listdir(path):
        if file.endswith(".csv"):
            model_name = file.split("_")[0]
            data = pd.read_csv(os.path.join(path, file))
            plt.plot(data["n_features"], data[score], label=model_name, marker='o')

    plt.title(title)
    plt.xticks(data["n_features"])
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{title.replace(" ", "_").lower()}.png")
