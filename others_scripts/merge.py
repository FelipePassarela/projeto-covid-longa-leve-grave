import pandas as pd
import os


def merge(analises_path: str, geral_path: str, save_path: str, column_name: str):
    """
    Merge the analysis CSV file with the general data CSV file and save the result.

    :param analises_path: The path to the analysis CSV file.
    :param geral_path: The path to the general data CSV file.
    :param save_path: The path to save the merged data.
    :param column_name: The name of the column to be merged.
    """
    try:
        if not os.path.exists(analises_path):
            raise FileNotFoundError("The analysis CSV file was not found.")
        if not os.path.exists(geral_path):
            raise FileNotFoundError("The general data CSV file was not found.")

        df_analises = pd.read_csv(analises_path)
        df_geral = pd.read_csv(geral_path)
        df_geral = df_geral.merge(df_analises[['patient_id', column_name]], on='patient_id', how='left')

        print(df_analises.head())
        print(df_geral.head())

        df_geral.to_csv(save_path, index=False)
        print("Data merged and saved successfully.")

    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    merge(
        "data/Planilha - GERAL.csv",
        "data/matriz_no_vac.csv",
        "data/risk_1/no_vac.csv",
        "Risk"
    )