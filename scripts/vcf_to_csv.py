#!/usr/bin/env python3
# This only works on linux and macOS

import argparse
from os import PathLike

import pandas as pd
from pysam import VariantFile


def vcf_to_csv(vcf_file: PathLike, csv_file: PathLike):
    """
    Convert a VCF file to a CSV file. The VCF file is expected to contain samples starting with "UFES".
    The CSV file will have the following structure:
    - The first column will contain the sample IDs.
    - The subsequent columns will contain the genotype information for each sample.

    :param vcf_file: Path to the input VCF file.
    :param csv_file: Path to the output CSV file.
    """
    print(f"Converting VCF file {vcf_file} to CSV file {csv_file}")

    with VariantFile(vcf_file) as vcf:
        sample_dict = {sample: [] for sample in vcf.header.samples}

        assert len(sample_dict) > 0, "No samples found in the VCF file."
        assert all(sample.startswith("UFES") for sample in sample_dict), "Not all samples start with 'UFES'"

        headers = []

        for i, rec in enumerate(vcf.fetch()):
            print(f"Processing record {i + 1}, chr{rec.chrom}:{rec.pos}".ljust(80), end='\r')
            headers.append(f"chr{rec.chrom}_{rec.pos}")

            for sample in rec.samples:
                genotype = rec.samples[sample]['GT']
                match genotype:
                    case (0, 0):
                        sample_dict[sample].append(0)
                    case (0, 1) | (1, 0):
                        sample_dict[sample].append(1)
                    case (1, 1):
                        sample_dict[sample].append(2)
                    case _:
                        sample_dict[sample].append(None)
        print("\nFinished processing records.")

        assert len(headers) > 0, "No headers found in the VCF file."
        assert len(sample_dict) > 0, "No samples found in the VCF file."
        assert all(len(sample_dict[sample]) == len(headers) for sample in sample_dict), \
            "Sample data length mismatch with headers"

        # float64 is used to handle NaN values
        df = pd.DataFrame.from_dict(sample_dict, orient='index', columns=headers, dtype='float64')
        df.insert(0, 'id', df.index)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(csv_file, index=False)

        print("First 5 rows of the DataFrame:")
        print(df.head())

        print(f"DataFrame created with shape {df.shape}")
        print(f"CSV file saved as {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert VCF file to CSV format')
    parser.add_argument('vcf', help='Input VCF file path')
    parser.add_argument('csv', help='Output CSV file path')
    args = parser.parse_args()
    
    vcf_to_csv(args.vcf, args.csv)