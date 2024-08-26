import os
import pandas as pd
import matplotlib.pyplot as plt

SCORE = "roc_auc"
SCORE_MATCH = {
    "accuracy": "Acurácia",
    "f1": "F1",
    "roc_auc": "ROC AUC"
}

def main():
    plt.figure(figsize=(10, 6))

    for file in os.listdir("results"):
        if file.endswith(".csv"):
            model_name = file.split("_")[0]
            data = pd.read_csv(os.path.join("results", file))
            plt.plot(data["n_features"], data[SCORE], label=model_name, marker='o')

    score_title = SCORE_MATCH[SCORE]

    plt.title(f"{score_title} para Diferentes Modelos")
    plt.xticks(data["n_features"])
    plt.xlabel("Número de Variantes (num_snp)")
    plt.ylabel(score_title)
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/new_plot_{SCORE}.png")
    plt.show()

if __name__ == "__main__":
    main()
