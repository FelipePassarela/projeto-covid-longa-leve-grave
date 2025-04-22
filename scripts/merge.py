import os

import pandas as pd


def merge(
        spreadsheet_path: str, 
        genom_path: str, 
        merged_path: str, 
        target_column: str,
    ) -> None:
    """
    Merge the genomic data with the patient information data.

    :param spreadsheet_path: The path to the patient information data.
    :param genom_path: The path to the genomic data.
    :param merged_path: The path to save the merged data.
    :param target_column: The name of the target column to merge on.
    """
    try:
        if not os.path.exists(spreadsheet_path):
            raise FileNotFoundError(f"File {spreadsheet_path} not found.")
        if not os.path.exists(genom_path):
            raise FileNotFoundError(f"File {genom_path} not found.")

        df_info = pd.read_csv(spreadsheet_path, usecols=['id', target_column])
        df_genom = pd.read_csv(genom_path)

        format_ids(df_info, df_genom)

        df_merged = pd.merge(df_genom, df_info, on='id', how='left')
        df_merged = df_merged.dropna(subset=[target_column])

        missing_ids = set(df_info['id']) - set(df_merged['id'])
        print("Missing IDs in genomic dataset:", missing_ids)

        df_merged.to_csv(merged_path, index=False)
        print("Data merged and saved successfully.")

    except FileNotFoundError as e:
        print(e)


def format_ids(df_info: pd.DataFrame, df_genom: pd.DataFrame) -> None:
    """
    Format the IDs in the dataframes to ensure they match.

    :param df_info: The patient information dataframe.
    :param df_genom: The genomic data dataframe.    
    """
    df_info['id'] = df_info['id'].astype(str).str.strip()
    df_genom['id'] = df_genom['id'].astype(str).str.strip()
    df_genom['id'] = df_genom['id'].str.replace("UFES_", "")
    df_genom['id'] = df_genom['id'].str.lstrip("0")
    df_genom['id'] = df_genom['id'].str.replace("-", "_")


if __name__ == "__main__":
    genomic_path = "data/genomics/matriz_TESTE_2_AGORA.csv"
    datasets = [
        ("data/grave/geral/planilha.csv", genomic_path, "data/grave/geral/merged.csv", "risk"),
        ("data/grave/nao_vacinados/planilha.csv", genomic_path, "data/grave/nao_vacinados/merged.csv", "risk"),

        ("data/longa/geral/planilha.csv", genomic_path, "data/longa/geral/merged.csv", "Long_COVID"),
        ("data/longa/nao_vacinados/planilha.csv", genomic_path, "data/longa/nao_vacinados/merged.csv", "Long_COVID"),

        ("data/dor/nao_vacinados/planilha.csv", genomic_path, "data/dor/nao_vacinados/merged.csv", "Pain_Block"),
        ("data/sistema_cardiovascular/nao_vacinados/planilha.csv", genomic_path, "data/sistema_cardiovascular/nao_vacinados/merged.csv", "Cardiovascular_sequelae"),
        ("data/sistema_nervoso/nao_vacinados/planilha.csv", genomic_path, "data/sistema_nervoso/nao_vacinados/merged.csv", "Sist_Nerv_Per_"),
    ]
    
    for spreadsheet_path, genom_path, save_path, target_column in datasets:
        print(f"Merging {spreadsheet_path} with {genom_path} on column {target_column}")
        merge(spreadsheet_path, genom_path, save_path, target_column)