#!/usr/bin/env python3


import argparse
import gc
import glob
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd


def get_columns_and_types(thisdf) -> Dict[str, List[str]]:
    columns = [c for c in thisdf.columns if not c.startswith("an_")]

    baseline_columns = [
        c
        for c in columns
        if c.startswith("baseline") and "head" not in c and "tail" not in c
    ]
    baseline_columns = [c for c in baseline_columns if "filesize" not in c]
    baseline_columns = [
        c for c in baseline_columns if "begin" not in c and "end" not in c
    ]

    advanced_columns = [c for c in columns if "advanced" in c]
    advanced_columns = [
        c for c in advanced_columns if "begin" not in c and "end" not in c
    ]
    advanced_columns = [
        c for c in advanced_columns if "head" not in c and "tail" not in c
    ]
    advanced_columns = [c for c in advanced_columns if "start" not in c]
    advanced_columns_only = list(set(advanced_columns))
    advanced_columns = list(set(advanced_columns + baseline_columns))

    fourier_columns = [c for c in columns if "fourier" in c and "value" not in c]
    fourier_columns = [c for c in fourier_columns if "1byte" in c]
    fourier_columns = [
        c for c in fourier_columns if "begin" not in c and "end" not in c
    ]
    fourier_columns = [
        c for c in fourier_columns if "head" not in c and "tail" not in c
    ]
    fourier_columns = [c for c in fourier_columns if "start" not in c]
    fourier_columns_only = list(set(fourier_columns))
    fourier_columns = list(set(advanced_columns + fourier_columns))

    baseline_and_advanced = list(set(baseline_columns + advanced_columns_only))
    baseline_and_fourier = list(set(baseline_columns + fourier_columns_only))
    advanced_and_fourier = list(set(advanced_columns_only + fourier_columns_only))

    return {
        "baseline": baseline_columns,
        "advanced-only": advanced_columns_only,
        "fourier-only": fourier_columns_only,
        "baseline-and-fourier": baseline_and_fourier,
        "baseline-and-advanced": baseline_and_advanced,
        "advanced-and-fourier": advanced_and_fourier,
        "advanced": advanced_columns,
        "fourier": fourier_columns,
    }


def get_annotation_columns(thisdf) -> List(str):
    return [c for c in thisdf.columns if c.startswith("an_")]


def annotate_df_with_additional_fields(
    name: str, dataframe: pd.DataFrame
) -> pd.DataFrame:
    """Add some metadata to each dataframe

    Args:
        name (str): Name of the csv/parquet file
        dataframe (pd.DataFrame): Dataframe

    Returns:
        pd.DataFrame: Dataframe with additional information
    """
    if "base32" in name:
        dataframe["base32"] = 1
    else:
        dataframe["base32"] = 0
    dataframe["base32"] = dataframe["base32"].astype(np.int8)
    if "encrypted" in name:
        dataframe["encrypted"] = 1
    else:
        dataframe["encrypted"] = 0
    dataframe["encrypted"] = dataframe["encrypted"].astype(np.int8)
    if "_v0" in name:
        dataframe["an_v0_encrypted"] = 1
    else:
        dataframe["an_v0_encrypted"] = 0
    dataframe["an_v0_encrypted"] = dataframe["an_v0_encrypted"].astype(np.int8)
    if "_v1" in name:
        dataframe["an_v1_encrypted"] = 1
    else:
        dataframe["an_v1_encrypted"] = 0
    dataframe["an_v1_encrypted"] = dataframe["an_v1_encrypted"].astype(np.int8)
    if "_v2" in name:
        dataframe["an_v2_encrypted"] = 1
    else:
        dataframe["an_v2_encrypted"] = 0
    dataframe["an_v2_encrypted"] = dataframe["an_v2_encrypted"].astype(np.int8)
    return dataframe


def load_data(input_directory: str) -> pd.DataFrame:
    """Load all pandas data files from a directory and annotate them with
    additional fields

    Args:
        input_directory (str): input directory

    Returns:
        pd.DataFrame: A combined dataframe of all files
    """
    p = 0.01
    dataframes = {
        f: pd.read_csv(f, skiprows=lambda i: i > 0 and random.random() > p)
        for f in glob.glob(f"{input_directory}/*.csv")
    }
    dataframes = {
        n: annotate_df_with_additional_fields(n, df) for n, df in dataframes.items()
    }
    for n, df in dataframes.items():
        break
    _ = [gc.collect(i) for i in range(3) for j in range(3) for k in range(3)]
    return df


def main() -> None:
    parser = argparse.ArgumentParser("Run experiments")
    parser.add_argument("-i", "--input-directory", type=str, required=True)
    parser.add_argument("-o", "--output-directory", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_directory) or not os.path.isdir(
        args.input_directory
    ):
        raise Exception(f"Path {args.input_directory} does not exist")
    if not os.path.exists(args.output_directory) or not os.path.isdir(
        args.output_directory
    ):
        raise Exception(f"Path {args.output_directory} does not exist")

    data = load_data(args.input_directory)


if "__main__" == __name__:
    main()
