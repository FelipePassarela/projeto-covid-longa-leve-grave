import numpy as np
from umap import UMAP
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from utils.preprocessing import load_data, preprocess_data
import pickle


def umap_plot():
    df = load_data("final_genotipos_GERAL_RISK.csv")

    for i in range(5, 51, 5):
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
        plt.title(f"Projeção UMAP dos dados\n({selector.n_features_} características selecionadas)")
        plt.gca().set_aspect('equal', 'datalim')
        plt.grid(False)

        handle, _ = scatter.legend_elements()
        plt.legend(handle, ["Grave", "Leve"], title="Risco")

        plt.tight_layout()
        plt.savefig(f"plots/umap/UMAP_{selector.n_features_}feats.png")
        plt.clf()
        plt.close()

if __name__ == "__main__":
    umap_plot()