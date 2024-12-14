import pandas as pd
import os


def merge():
    """
    Merges data from two CSV files and saves the result.
    This function reads two CSV files: one containing general data and another containing analysis data.
    It merges the analysis data into the general data based on the 'patient_id' column and saves the merged
    data back to the general data CSV file.

    Raises:
        FileNotFoundError: If either of the CSV files is not found.
    Prints:
        The first few rows of the analysis data and the merged data.
        A success message upon successful merge and save.
    """

    try:
        analises_path = "../data/risk/analises - GERAL.csv"
        geral_path = "../data/risk/matriz_genotipos_no_vac_filtrado.csv"

        if not os.path.exists(analises_path):
            raise FileNotFoundError("O arquivo CSV de análises não foi encontrado.")
        if not os.path.exists(geral_path):
            raise FileNotFoundError("O arquivo CSV de dados gerais não foi encontrado.")

        df_analises = pd.read_csv(analises_path)
        df_geral = pd.read_csv(geral_path)
        df_geral = df_geral.merge(df_analises[['patient_id', 'risk']], on='patient_id', how='left')

        print(df_analises.head())
        print(df_geral.head())

        df_geral.to_csv(geral_path, index=False)
        print("Merge realizado e arquivo salvo com sucesso.")

    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    merge()