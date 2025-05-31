import argparse
from pathlib import Path
import logging

import pandas as pd


def filter_variants(csv_genomic: Path, csv_variant: Path) -> pd.DataFrame:
    """
    Filter variants from a genomic CSV file based on a variant CSV file.

    :param csv_genomic: Path to the genomic CSV file.
    :param csv_variant: Path to the variant CSV file.
    :return: DataFrame containing filtered variants.
    """
    logging.info(f"Loading genomic data from {csv_genomic}")
    df_genomic = pd.read_csv(csv_genomic)

    logging.info(f"Loading variant data from {csv_variant}")
    df_variant = pd.read_csv(csv_variant)

    df_variants_to_keep = "chr" + df_variant["Chr"].astype(str) + "_" + df_variant["Start"].astype(str)
    total_variants = sum(col.startswith("chr") for col in df_genomic.columns)
    logging.info(f"Found {len(df_variants_to_keep)} variants in the variant file.")
    logging.info(f"Found {total_variants} variants in the genomic file.")

    columns_to_keep = [col for col in df_genomic.columns if not col.startswith("chr")]
    existing_variants_to_keep = [var for var in df_variants_to_keep if var in df_genomic.columns]
    columns_to_keep += existing_variants_to_keep
    
    if len(existing_variants_to_keep) < len(df_variants_to_keep):
        n_missing_variants = len(df_variants_to_keep) - len(existing_variants_to_keep)
        logging.warning(f"{n_missing_variants} variants from the variant file were"
                        f" not found in the genomic file and will be ignored.")

    logging.info(f"Keeping {len(columns_to_keep)} columns ({len(existing_variants_to_keep)} variants) in the filtered DataFrame.")
    
    df_filtered = df_genomic[columns_to_keep]
    logging.info(f"Filtered DataFrame shape: {df_filtered.shape}")

    return df_filtered


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(description="Filter variants from a CSV file based on genomic data.")
    parser.add_argument("csv_genomic", type=Path, help="CSV file containing genomic data with variant columns to be filtered.")
    parser.add_argument("csv_variant", type=Path, help="CSV file containing the list of variants to keep.")
    
    args = parser.parse_args()

    if not args.csv_genomic.exists():
        raise FileNotFoundError(f"Genomic CSV file {args.csv_genomic} does not exist.")
    if not args.csv_variant.exists():
        raise FileNotFoundError(f"Variant CSV file {args.csv_variant} does not exist.")
    
    filtered_df = filter_variants(args.csv_genomic, args.csv_variant)

    output_path = args.csv_variant.with_name(f"matriz_{args.csv_variant.name}")
    filtered_df.to_csv(output_path, index=False)
    logging.info(f"Filtered variants saved to {output_path}")
