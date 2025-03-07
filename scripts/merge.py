import os

import pandas as pd


def merge(info_path: str, genom_path: str, save_path: str, column_name: str):
    """
    Merge the genomic data with the patient information data.

    :param info_path: The path to the patient information data.
    :param genom_path: The path to the genomic data.
    :param save_path: The path to save the merged data.
    :param column_name: The name of the column to merge on.
    """
    try:
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"File {info_path} not found.")
        if not os.path.exists(genom_path):
            raise FileNotFoundError(f"File {genom_path} not found.")

        df_info = pd.read_csv(info_path)
        df_genom = pd.read_csv(genom_path)
        df_genom = df_genom.merge(df_info[['patient_id', column_name]], on='patient_id', how='left')

        print(df_info.head())
        print(df_genom.head())

        df_genom.to_csv(save_path, index=False)
        print("Data merged and saved successfully.")

    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    merge(
        "data/mion/geral/Bloco Dor - GERAL.csv",
        "data/mion/geral/MATRIZ_GERAL_FILTRADO.csv",
        "data/mion/geral/MATRIZ_GERAL_FILTRADO_merged.csv",
        "Pain Block_175"
    )