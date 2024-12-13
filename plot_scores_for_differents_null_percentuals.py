from matplotlib import pyplot as plt
import pandas as pd
import os


def histogram_null_values(df: pd.DataFrame) -> None:
    """
    Plot a histogram of the percentage of missing values in the columns of the 
    dataframe and indicate the 95th and 99th percentiles.

    :param df: The dataframe with the data.
    :type df: pd.DataFrame
    """
    missing_percentage = df.isnull().mean() * 100
    missing_percentage.hist(bins=10, figsize=(10, 6))

    title = "Histograma de valores faltantes nas colunas"
    plt.title(title)
    plt.xlabel("Porcentagem de valores faltantes")
    plt.ylabel("Número de colunas")

    percentile_95 = missing_percentage.quantile(0.95)
    plt.axvline(percentile_95, color='r', linestyle='dashed', linewidth=1)
    plt.text(percentile_95 * 1.01, plt.ylim()[1] * 0.95, f'95th Percentile: {percentile_95:.2f}%', color='r')

    percentile_99 = missing_percentage.quantile(0.99)
    plt.axvline(percentile_99, color='r', linestyle='dashed', linewidth=1)
    plt.text(percentile_99 * 1.01, plt.ylim()[1] * 0.95, f'99th Percentile: {percentile_99:.2f}%', color='r')

    plt.savefig(f"backup/{title.replace(' ', '_')}.png")
    plt.show()


def plot_score_for_differents_null_thresholds(metric: str, n_features: int) -> None:
    df_to_plot = pd.DataFrame(columns=["threshold", "score", "clf", "tuned"])

    for directory in os.listdir("backup/"):
        if not directory.startswith("backup_"):
            continue

        threshold = float(directory.split("_")[1].replace("%", ""))
        best_clf_of_threshold = {
            "clf": "",
            "score": 0,
            "tuned": False
        }
        
        for classifier_csv in os.listdir(f"backup/{directory}/results/test/standard"):
            if not classifier_csv.endswith(".csv"):
                continue

            df_standard = pd.read_csv(f"backup/{directory}/results/test/standard/{classifier_csv}")
            df_tuned = pd.read_csv(f"backup/{directory}/results/test/tuned/{classifier_csv}")
            
            score_standard = df_standard[df_standard["n_features"] == n_features][metric].values[0]
            score_tuned = df_tuned[df_tuned["n_features"] == n_features][metric].values[0]

            if max(score_standard, score_tuned) > best_clf_of_threshold["score"]:
                best_clf_of_threshold["score"] = max(score_standard, score_tuned)
                best_clf_of_threshold["clf"] = classifier_csv.split(".")[0]
                best_clf_of_threshold["tuned"] = score_tuned > score_standard

        new_row = pd.DataFrame({
            "threshold": [threshold],
            "score": [best_clf_of_threshold["score"]],
            "clf": [best_clf_of_threshold["clf"]],
            "tuned": [best_clf_of_threshold["tuned"]]
        })
        df_to_plot = pd.concat([df_to_plot, new_row], ignore_index=True)

    df_to_plot = df_to_plot.sort_values(by="threshold")

    METRIC_NAME_MATCH = {
        "roc_auc": "ROC AUC",
        "f1": "F1",
        "accuracy": "Acurácia",
    }
    CLF_NAME_MATCH = {
        "LogisticRegression": "LR",
        "XGBClassifier": "XGB",
        "RandomForestClassifier": "RF",
    }

    plt.figure(figsize=(10, 6))
    plt.plot(df_to_plot["threshold"], df_to_plot["score"], marker="o")
    plt.xlabel("Porcentagem de valores faltantes")
    plt.ylabel(METRIC_NAME_MATCH[metric])

    # Increase the axis scale to make the text fit
    plt.xlim(min(df_to_plot["threshold"]) - 0.5, max(df_to_plot["threshold"]) + 1.5)
    plt.ylim(min(df_to_plot["score"]) - 0.05, max(df_to_plot["score"]) + 0.02)

    title = f"{METRIC_NAME_MATCH[metric]} para Diferentes Porcentagens de Valores Faltantes ({n_features} SNPs)"
    plt.title(title)
    plt.grid()

    for _, row in df_to_plot.iterrows():
        tuned_str = "(Otimizado)" if row["tuned"] else ""
        clf_name = CLF_NAME_MATCH[row["clf"]] if row["clf"] in CLF_NAME_MATCH else row["clf"]
        plt.text(row["threshold"] + 0.1, row["score"], f"{clf_name}{tuned_str}", va="bottom", ha="left")

    plt.tight_layout()
    plt.savefig(f"backup/{title.replace(' ', '_')}.png")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("final_genotipos_GERAL_RISK.csv")
    histogram_null_values(df)
    plot_score_for_differents_null_thresholds("roc_auc", 10)
    plot_score_for_differents_null_thresholds("roc_auc", 15)
