import argparse
from functools import partial
from typing import Dict

import pandas as pd
from sklearn import metrics

stats = {
    "Accuracy": (
        lambda df: metrics.accuracy_score(df["y_true"], df["y_pred"])
    ),
    "Balanced-Accuracy": (
        lambda df: metrics.balanced_accuracy_score(df["y_true"], df["y_pred"])
    ),
    "Precision": (
        lambda df: metrics.precision_score(df["y_true"], df["y_pred"])
    ),
    "Recall": (lambda df: metrics.recall_score(df["y_true"], df["y_pred"])),
    "F1-Score": (lambda df: metrics.f1_score(df["y_true"], df["y_pred"])),
    "AUROC": (
        lambda df: metrics.roc_auc_score(df["y_true"], df["y_pred_proba"])
    ),
}


def get_order_number(name: str) -> int:
    order_of_columns = [
        "baseline-only",
        "advanced-only",
        "fourier-only",
        "baseline-and-advanced",
        "baseline-and-fourier",
        "advanced-and-fourier",
        "baseline-advanced-and-fourier",
    ]

    for i, n in enumerate(order_of_columns):
        if name == n:
            return i
    return -1


def get_combined_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            fnname: df.groupby(["feature_set"])[
                ["y_true", "y_pred", "y_pred_proba"]
            ].apply(fn)
            for fnname, fn in stats.items()
        }
    ).reset_index()
    df["order"] = df["feature_set"].map(get_order_number)
    df = df.sort_values(by="order", ignore_index=True)
    df.drop("order", axis=1, inplace=True)
    return df


def get_grouped_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            fnname: df.groupby(["feature_set", "run_name"])[
                ["y_true", "y_pred", "y_pred_proba"]
            ].apply(fn)
            for fnname, fn in stats.items()
        }
    ).reset_index()
    columns = [k for k, _ in stats.items()]
    df = df.groupby("feature_set")[columns].agg(["mean", "std"]).reset_index()
    df["order"] = df["feature_set"].map(get_order_number)
    df = df.sort_values(by="order", ignore_index=True)
    df = df.set_index("feature_set")
    df.drop("order", axis=1, inplace=True)
    return df.reset_index()


def get_metrics_comparisons(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    combined_stats = get_combined_stats(df)
    grouped_stats = get_grouped_stats(df)
    return {"combined_stats": combined_stats, "grouped_stats": grouped_stats}


def print_latex(
    df: pd.DataFrame, highlight_min_max: bool = False, num_decimals: int = 3
) -> None:
    def format_min_max(
        x,
        min_value: float,
        max_value: float,
        num_decimals: int = 3,
        invert: bool = False,
    ) -> str:
        if isinstance(x, str):
            return x
        # if round(x, num_decimals) == round(min_value, num_decimals):
        if x == min_value:
            if not invert:
                return f"\\textOrange{{{x:.{num_decimals}f}}}"
            else:
                return f"\\textBlue{{{x:.{num_decimals}f}}}"
        # elif round(x, num_decimals) == round(max_value, num_decimals):
        elif x == max_value:
            if not invert:
                return f"\\textBlue{{{x:.{num_decimals}f}}}"
            else:
                return f"\\textOrange{{{x:.{num_decimals}f}}}"
        else:
            return f"{x:.{num_decimals}f}"

    formatters = {
        colname: partial(
            format_min_max,
            min_value=df[colname].min(),
            max_value=df[colname].max(),
            num_decimals=num_decimals,
            invert=("std" in colname and isinstance(colname, tuple)),
        )
        for colname in df.columns
        if colname != ("feature_set", "")
    }

    # Get rid of some columns that we don't want to print
    df = df[
        [
            c
            for c in df.columns
            if (
                not isinstance(c, tuple)
                or c[0] not in ["Balanced-Accuracy", "Accuracy"]
            )
        ]
    ]

    for c in df.columns:
        print(c, type(c))

    print()
    if highlight_min_max:
        df_str = df.to_latex(formatters=formatters, escape=False)
    else:
        df_str = df.to_latex()
    print(df_str)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True)
    parser.add_argument(
        "-tl", "--to-latex", action="store_true", default=False
    )
    parser.add_argument(
        "-hm", "--highlight-min-max", action="store_true", default=False
    )
    parser.add_argument("-nd", "--num-decimals", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    comparisons = get_metrics_comparisons(df)
    print("COMBINED")
    print("-" * len("COMBINED"))
    print(comparisons["combined_stats"].round(3).reset_index(drop=True))
    if args.to_latex:
        print_latex(
            comparisons["combined_stats"].reset_index(drop=True),
            args.highlight_min_max,
            args.num_decimals,
        )
    print("\n\n\nGROUPED")
    print("-" * len("GROUPED"))
    print(comparisons["grouped_stats"].round(3).reset_index(drop=True))
    if args.to_latex:
        print_latex(
            comparisons["grouped_stats"].reset_index(drop=True),
            args.highlight_min_max,
            args.num_decimals,
        )


if "__main__" == __name__:
    main()
