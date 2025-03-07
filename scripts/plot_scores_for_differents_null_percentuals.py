import os

import pandas as pd
from matplotlib import pyplot as plt


def histogram_null_values(df: pd.DataFrame) -> None:
    """
    Plot a histogram of the percentage of missing values in the columns of the 
    dataframe and indicate the 95th and 99th percentiles.

    :param df: The dataframe with the data.
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

    path = f"output/{title.replace(' ', '_')}.png"
    os.makedirs("output", exist_ok=True)
    plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/matriz_GERAL.csv")
    histogram_null_values(df)
