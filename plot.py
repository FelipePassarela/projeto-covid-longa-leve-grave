import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    plt.figure(figsize=(10, 6))

    for file in os.listdir("results"):
        if file.endswith(".csv"):
            model_name = file.split("_")[0]
            data = pd.read_csv(os.path.join("results", file))
            plt.plot(data["n_features"], data["roc_auc"], label=model_name, marker='o')

    plt.title("AUC ROC para Diferentes Modelos")
    plt.xticks(data["n_features"])
    plt.xlabel("Número de Variantes (num_snp)")
    plt.ylabel("AUC ROC")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()

    plt.savefig("plot.png")
    plt.show()

if __name__ == "__main__":
    main()

