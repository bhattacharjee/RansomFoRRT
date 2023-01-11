#!/usr/bin/env python3


import argparse
import copy
import gc
import glob
import json
import os
import random
import sys
import time
import uuid
import warnings
from typing import Callable, Dict, List, Tuple
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tqdm
import shap
from loguru import logger
from sklearn import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

# import dotenv


# Way to dynamically change the number of jobs at run time
def get_num_jobs(default_jobs: int) -> int:
    """This function provides a way to override the number of jobs specified
    in the command line arguments dynamically.
    A file called num_jobs.txt can be created and the first line
    should contain the number of jobs.

    Args:
        default_jobs (int): default value if it is not overridden

    Returns:
        int: number of jobs to run
    """
    if not os.path.exists("num_jobs.txt"):
        return default_jobs
    with open("num_jobs.txt") as f:
        try:
            line = f.readlines()[0].strip()
            temp_jobs = int(line)
            if temp_jobs > 0 and temp_jobs < 20:
                logger.info(f"NUM_JOBS override: {temp_jobs}")
                return temp_jobs
        except:
            return default_jobs
    return default_jobs


def random_seed() -> None:
    np.random.seed(0)
    random.seed(0)


def gc_collect() -> None:
    for i in range(3):
        for j in range(3):
            gc.collect(j)


def get_columns_and_types(thisdf: pd.DataFrame) -> Dict[str, List[str]]:
    """For each feature set type, get the relevant columns.

    Args:
        thisdf (pd.DataFrame): Input dataframe.

    Returns:
        Dict[str, List[str]]: Dictionary that maps the feature type to the
            list of columns to the feature type.
    """
    columns = [c for c in thisdf.columns if not c.startswith("an_")]

    def get_columns(columns: List[str], start_string: str) -> List[str]:
        columns = [c for c in columns if c.startswith(start_string)]
        columns = [c for c in columns if "head" not in c and "tail" not in c]
        columns = [c for c in columns if "begin" not in c and "end" not in c]
        columns = [c for c in columns if "filesize" not in c]
        return columns

    baseline_columns = get_columns(columns, "baseline")
    advanced_columns = get_columns(columns, "advanced")
    fourier_columns = get_columns(columns, "fourier")
    fourier_min_columns = [
        "fourier.stat.1byte.autocorr",
        "fourier.stat.1byte.mean",
        "fourier.stat.1byte.std",
        "fourier.stat.1byte.chisq",
        "fourier.stat.1byte.moment.2",
        "fourier.stat.1byte.moment.3",
        "fourier.stat.1byte.moment.4",
        "fourier.stat.1byte.moment.5",
    ]

    baseline_and_advanced = list(set(baseline_columns + advanced_columns))
    baseline_and_fourier = list(set(baseline_columns + fourier_columns))
    advanced_and_fourier = list(set(advanced_columns + fourier_columns))
    baseline_and_fourier_min = list(
        set(baseline_columns + fourier_min_columns)
    )
    advanced_and_fourier_min = list(
        set(advanced_columns + fourier_min_columns)
    )

    baseline_advanced_fourier = list(
        set(baseline_columns + advanced_columns + fourier_columns)
    )
    baseline_advanced_and_fourier_min = list(
        set(baseline_columns + advanced_columns + fourier_min_columns)
    )

    rv = {
        #"baseline-only": baseline_columns,
        #"advanced-only": advanced_columns,
        #"fourier-only": fourier_columns,
        #"fourier-min-only": fourier_min_columns,
        #"baseline-and-fourier": baseline_and_fourier,
        #"baseline-and-fourier-min": baseline_and_fourier_min,
        #"baseline-and-advanced": baseline_and_advanced,
        #"advanced-and-fourier": advanced_and_fourier,
        #"advanced-and-fourier-min": advanced_and_fourier_min,
        "baseline-advanced-and-fourier": baseline_advanced_fourier,
        #"baseline-advanced-and-fourier-min": baseline_advanced_and_fourier_min,
    }

    logger.info(f"Features = {rv}")

    return rv["baseline-advanced-and-fourier"]


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
    if "base32" in name or "b32" in name:
        dataframe["an_is_base32"] = 1
    else:
        dataframe["an_is_base32"] = 0
    dataframe["an_is_base32"] = dataframe["an_is_base32"].astype(np.bool_)

    if "encrypt" in name:
        dataframe["is_encrypted"] = 1
    else:
        dataframe["is_encrypted"] = 0
    dataframe["is_encrypted"] = dataframe["is_encrypted"].astype(np.bool_)

    if "v1" in name:
        dataframe["an_v1_encrypted"] = 1
    else:
        dataframe["an_v1_encrypted"] = 0
    dataframe["an_v1_encrypted"] = dataframe["an_v1_encrypted"].astype(
        np.bool_
    )

    if "v2" in name:
        dataframe["an_v2_encrypted"] = 1
    else:
        dataframe["an_v2_encrypted"] = 0
    dataframe["an_v2_encrypted"] = dataframe["an_v2_encrypted"].astype(
        np.bool_
    )

    if "v3" in name:
        dataframe["an_v3_encrypted"] = 1
    else:
        dataframe["an_v3_encrypted"] = 0
    dataframe["an_v3_encrypted"] = dataframe["an_v3_encrypted"].astype(
        np.bool_
    )

    def is_webp(filename: str) -> int:
        return 1 if ".webp" in filename else 0

    dataframe["an_is_webp"] = (
        dataframe["extended.base_filename"].map(is_webp).astype(np.bool_)
    )

    return dataframe


def load_data(input_directory: str) -> pd.DataFrame:
    """Load all pandas data files from a directory and annotate them with
    additional fields

    Args:
        input_directory (str): input directory

    Returns:
        pd.DataFrame: A combined dataframe of all files
    """
    p = 0.1
    logger.info("Loading dataframes")
    dataframes = {
        #f: pd.read_csv(f, skiprows=lambda i: i > 0 and random.random() > p)
        f: pd.read_csv(f)
        for f in glob.glob(f"{input_directory}{os.path.sep}*.csv.gz")
    }
    logger.info("Annotating dataframes with additional fields")
    dataframes = {
        f: annotate_df_with_additional_fields(f, df)
        for f, df in dataframes.items()
    }

    logger.info("Combining dataframes into a single dataframe")
    df = (
        pd.concat([df for _, df in dataframes.items()])
        .sample(frac=1)
        .reset_index(drop=True)
    )

    gc_collect()

    logger.info("done...")
    return df


def get_pipeline(
    X: pd.DataFrame, n_jobs: int = 4
) -> Tuple[pipeline.Pipeline, Callable[[np.array], np.array]]:
    random_seed()
    num_jobs = get_num_jobs(n_jobs)
    pipe = pipeline.Pipeline(
        [
            ("std", MinMaxScaler()),
            ("classif", RandomForestClassifier(n_jobs=num_jobs)),
        ]
    )
    # In case the predicted value needs to be converted to an integer
    # this lambda will do the work
    return pipe, lambda x: x


def get_metrics(y_true: np.array, y_pred: np.array) -> List[float]:
    def error_checked_metric(fn, y_true, y_pred):
        try:
            return fn(y_true, y_pred)
        except Exception as e:
            return 1.0  # Treat as fully accurate

    return [
        error_checked_metric(fn, y_true, y_pred)
        for fn in [
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        ]
    ]


def trim_dataset(
    df: pd.DataFrame,
    exclude_plaintext_nonbase32: bool = False,
    exclude_plaintext_base32: bool = False,
    exclude_encrypted_v1: bool = False,
    exclude_encrypted_v2: bool = False,
    exclude_encrypted_base32: bool = False,
    exclude_encrypted_nonbase32: bool = False,
    exclude_webp: bool = False,
    exclude_nonwebp: bool = False,
):
    df = df.copy()
    logger.debug(f"0 ===> {len(df)}")

    # This is not a realistic combination and the caller should never
    # call this. Putting this assert in place for debugging reasons.
    assert exclude_plaintext_nonbase32 == False

    if exclude_plaintext_nonbase32:
        selector = ~(df["is_encrypted"].astype(np.bool_)) & ~(
            df["an_is_base32"].astype(np.bool_)
        )
        df = df[~selector]
    logger.debug(f"1 ===> {len(df)}")

    if exclude_plaintext_base32:
        selector = ~(df["is_encrypted"].astype(np.bool_)) & df[
            "an_is_base32"
        ].astype(np.bool_)
        df = df[~selector]
    logger.debug(f"2 ===> {len(df)}")

    if exclude_encrypted_v1:
        selector = df["an_v1_encrypted"].astype(np.bool_)
        df = df[~selector]
    logger.debug(f"3 ===> {len(df)}")

    if exclude_encrypted_v2:
        selector = df["an_v2_encrypted"].astype(np.bool_)
        df = df[~selector]
    logger.debug(f"4 ===> {len(df)}")

    if exclude_encrypted_base32:
        selector = df["is_encrypted"].astype(np.bool_) & df[
            "an_is_base32"
        ].astype(np.bool_)
        df = df[~selector]
    logger.debug(f"5 ===> {len(df)}")

    if exclude_encrypted_nonbase32:
        selector = df["is_encrypted"].astype(np.bool_) & ~(
            df["an_is_base32"].astype(np.bool_)
        )
        df = df[~selector]
    logger.debug(f"6 ===> {len(df)}")

    if exclude_webp:
        selector = df["an_is_webp"].astype(np.bool_)
        df = df[~selector]
    logger.debug(f"7 ===> {len(df)}")

    if exclude_nonwebp:
        selector = ~(df["an_is_webp"].astype(np.bool_))
        df = df[~selector]
    logger.debug(f"8 ===> {len(df)}")

    try:
        non_encrypted_count = (~df["is_encrypted"]).astype(np.int8).abs().sum()
    except:
        non_encrypted_count = 0
    try:
        encrypted_count = df["is_encrypted"].astype(np.int8).abs().sum()
    except:
        encrypted_count = 0

    logger.info(
        f"Encrypted: {encrypted_count} Non-Encrypted: {non_encrypted_count}"
    )

    gc_collect()

    if encrypted_count == 0 or non_encrypted_count == 0:
        return None

    return df


def main() -> None:
    # dotenv.load_dotenv()
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser("Run experiments")
    parser.add_argument(
        "-i",
        "--input-directory",
        type=str,
        required=True,
        help="Input directory for data files.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_directory) or not os.path.isdir(
        args.input_directory
    ):
        raise Exception(f"Path {args.input_directory} does not exist")
    logger.add("log.log", backtrace=True, diagnose=True, level="INFO")
    logger.add(f"log.debug.log", backtrace=True, diagnose=True, level="DEBUG")
    logger.add(sys.stderr, backtrace=True, diagnose=True, level="ERROR")
    logger.opt(colors=True).info(f"<blue>Running with {args}</>")

    random_seed()

    data = load_data(args.input_directory)

    annot_columns = get_annotation_columns(data)

    t1 = time.perf_counter()

    pline = get_pipeline(data)
    dataset = trim_dataset(data, *[False for i in range(8)])
    dataset = dataset[[c for c in dataset.columns if c not in annot_columns]]

    classif = RandomForestClassifier(n_jobs=-1)

    y = dataset["is_encrypted"].to_numpy()
    X = dataset[[c for c in dataset.columns if c != "is_encrypted"]]
    X = X[get_columns_and_types(X)].to_numpy()
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)
    classif.fit(X_train, y_train)

    X_test = X_test[:2000, :]

    explainer = shap.Explainer(classif.predict, X_test)
    shap_values = explainer(X_test, max_evals=1500)
    shap.plots.beeswarm(shap_values)
    plt.show()




if "__main__" == __name__:
    main()
