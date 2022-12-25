#!/usr/bin/env python3


import argparse
import copy
import gc
import glob
import os
import random
import time
import dotenv
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tqdm
from loguru import logger
from sklearn import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_columns_and_types(thisdf: pd.DataFrame) -> Dict[str, List[str]]:
    """For each feature set type, get the relevant columns.

    Args:
        thisdf (pd.DataFrame): Input dataframe.

    Returns:
        Dict[str, List[str]]: Dictionary that maps the feature type to the
            list of columns to the feature type.
    """
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

    fourier_columns = [
        c for c in columns if "fourier" in c and "value" not in c
    ]
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
    advanced_and_fourier = list(
        set(advanced_columns_only + fourier_columns_only)
    )

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


def get_annotation_columns(thisdf: pd.DataFrame) -> List[str]:
    """List of columns used for annotation.

    Args:
        thisdf (pd.DataFrame): Input dataframe.

    Returns:
        _type_: List of columns
    """
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
    if "is_encrypted" in name:
        dataframe["is_encrypted"] = 1
    else:
        dataframe["is_encrypted"] = 0
    dataframe["is_encrypted"] = dataframe["is_encrypted"].astype(np.int8)
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
        n: annotate_df_with_additional_fields(n, df)
        for n, df in dataframes.items()
    }
    for n, df in dataframes.items():
        break
    _ = [gc.collect(i) for i in range(3) for j in range(3) for k in range(3)]
    return df


def get_pipeline(X: pd.DataFrame, n_jobs: int = 4) -> pipeline.Pipeline:
    pipe = pipeline.Pipeline(
        [("classif", RandomForestClassifier(n_jobs=n_jobs))]
    )
    return pipe


def evaluate_features_folded(
    name: str,
    data: pd.DataFrame,
    output_directory: str,
    feature_column_names: List[str],
    annotation_columns: List[str],
    n_jobs: int,
    folds: int = -1,
) -> Tuple[bool, List[float]]:
    return True, []


def evaluate_features_regular(
    name: str,
    data: pd.DataFrame,
    output_directory: str,
    feature_column_names: List[str],
    annotation_columns: List[str],
    n_jobs: int,
) -> Tuple[bool, List[float]]:
    X = data[feature_column_names]
    y = data["is_encrypted"]
    pline = get_pipeline(X, n_jobs=n_jobs)
    return True, []


def evaluate_features(
    name: str,
    data: pd.DataFrame,
    output_directory: str,
    feature_column_names: List[str],
    annotation_columns: List[str],
    n_jobs: int,
    folds: int = -1,
) -> Tuple[bool, List[float]]:
    if folds != -1:
        return evaluate_features_folded(
            name=name,
            data=data,
            output_directory=output_directory,
            feature_column_names=feature_column_names,
            annotation_columns=annotation_columns,
            n_jobs=n_jobs,
            folds=folds,
        )
    else:
        return evaluate_features_regular(
            name=name,
            data=data,
            output_directory=output_directory,
            feature_column_names=feature_column_names,
            annotation_columns=annotation_columns,
            n_jobs=n_jobs,
        )


def evaluate(
    name: str,
    data: pd.DataFrame,
    output_directory: str,
    feature_column_names: List[str],
    annotation_columns: List[str],
    n_jobs: int,
    folds: int = -1,
) -> Tuple[bool, List[float]]:
    # This layer loops over the 54 different combinations
    return evaluate_features(
        name=name,
        data=data,
        output_directory=output_directory,
        feature_column_names=feature_column_names,
        annotation_columns=annotation_columns,
        n_jobs=n_jobs,
        folds=folds,
    )


def main() -> None:
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser("Run experiments")
    parser.add_argument(
        "-i",
        "--input-directory",
        type=str,
        required=True,
        help="Input directory for data files.",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "-n", "--n-jobs", type=int, default=4, help="Number of jobs to run."
    )
    parser.add_argument(
        "-nf", "--n-folds", type=int, default=-1, help="Folds to run for"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_directory) or not os.path.isdir(
        args.input_directory
    ):
        raise Exception(f"Path {args.input_directory} does not exist")
    if not os.path.exists(args.output_directory) or not os.path.isdir(
        args.output_directory
    ):
        raise Exception(f"Path {args.output_directory} does not exist")

    log_file = f"{args.output_directory}/log.log"
    if os.path.exists(log_file):
        os.unlink(log_file)
    logger.add(log_file, backtrace=True, diagnose=True)
    logger.opt(colors=True).info(f"<blue>Running with {args}</>")

    data = load_data(args.input_directory)

    annot_columns = get_annotation_columns(data)

    for n, (fsname, fscolumns) in enumerate(
        tqdm.tqdm(get_columns_and_types(data).items())
    ):
        temp_output_dir = f"{args.output_directory}/{fsname}"
        print_text = f"Processing {fsname} and writing into {temp_output_dir}"
        logger.opt(colors=True).info(f"<green>{print_text}</>")
        logger.opt(colors=True).info(f"<green>{'-' * len(print_text)}</>")

        columns = copy.copy(fscolumns)
        columns += annot_columns
        columns += ["is_encrypted"]

        if not os.path.exists(temp_output_dir):
            os.mkdir(temp_output_dir)
        t1 = time.perf_counter()
        logger.info(f"{n:02d}. Started evaluating feature set: {fsname}")
        retval, metrics = evaluate(
            name=fsname,
            data=data[columns].copy(),
            output_directory=temp_output_dir,
            feature_column_names=fscolumns,
            annotation_columns=annot_columns,
            n_jobs=args.n_jobs,
            folds=args.n_folds,
        )
        t2 = time.perf_counter()
        logger.info(
            f"{n:02d}. Completed running feature {fsname} in {t2 - t1} seconds"
        )
        logger.opt(colors=True).info(
            "<green>*******************************************************</>"
        )
        for i in range(3):
            for j in range(3): 
                gc.collect(i)
        if not retval:
            logger.error(
                f"Error evaluating feature set '{fsname}', metrics = {metrics}"
            )
            break
        logger.opt(colors=True).info(f"<magenta>{fsname} : {metrics}</>")


if "__main__" == __name__:
    main()
