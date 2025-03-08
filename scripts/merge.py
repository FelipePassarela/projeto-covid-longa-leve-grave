import os

import pandas as pd


def merge(info_path: str, genom_path: str, merged_path: str, column_name: str):
    """
    Merge the genomic data with the patient information data.

    :param info_path: The path to the patient information data.
    :param genom_path: The path to the genomic data.
    :param merged_path: The path to save the merged data.
    :param column_name: The name of the column to merge on.
    """
    try:
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"File {info_path} not found.")
        if not os.path.exists(genom_path):
            raise FileNotFoundError(f"File {genom_path} not found.")

        df_info = pd.read_csv(info_path, usecols=['id', column_name])
        df_genom = pd.read_csv(genom_path)

        # Ensure the id columns are equally formatted
        df_info['id'] = df_info['id'].astype(str).str.strip()
        df_genom['id'] = df_genom['id'].astype(str).str.strip()

        df_merged = pd.merge(df_genom, df_info, on='id', how='left')

        missing_ids = df_merged[df_merged[column_name].isnull()]['id']
        print("Missing IDs:", missing_ids)

        df_merged.to_csv(merged_path, index=False)
        print("Data merged and saved successfully.")

    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    datasets = [
        ("data/grave/geral/planilha.csv", "data/genom_processed.csv", "data/grave/geral/merged.csv", "risk"),
        ("data/grave/nao_vacinados/planilha.csv", "data/genom_processed.csv", "data/grave/nao_vacinados/merged.csv", "risk"),

        ("data/longa/geral/planilha.csv", "data/genom_processed.csv", "data/longa/geral/merged.csv", "Long_COVID"),
        ("data/longa/nao_vacinados/planilha.csv", "data/genom_processed.csv", "data/longa/nao_vacinados/merged.csv", "Long_COVID"),

        ("data/dor/nao_vacinados/planilha.csv", "data/genom_processed.csv", "data/dor/nao_vacinados/merged.csv", "Pain_Block"),
        ("data/sistema_cardiovascular/nao_vacinados/planilha.csv", "data/genom_processed.csv", "data/sistema_cardiovascular/nao_vacinados/merged.csv", "Cardiovascular_sequelae"),
        ("data/sistema_nervoso/nao_vacinados/planilha.csv", "data/genom_processed.csv", "data/sistema_nervoso/nao_vacinados/merged.csv", "Sist_Nerv_Per_"),
    ]
    
    for info_path, genom_path, save_path, column_name in datasets:
        print(f"Merging {info_path} with {genom_path} on column {column_name}")
        merge(info_path, genom_path, save_path, column_name)