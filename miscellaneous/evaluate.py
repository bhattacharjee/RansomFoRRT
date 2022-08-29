#!/usr/bin/env python3

import pandas as pd
import numpy as np
import seaborn as sns
import tqdm
import matplotlib.pyplot as plt


def get_threshold_array(df):
    df = df['correlation']
    out_df = {
        'threshold': [],
        'percentage': []
    }
    for i in tqdm.tqdm(range(1001)):
        j = 0 if i == 0 else i / 1000
        df2 = df > j
        n_up = np.sum(df2)
        n_all = df2.shape[0]
        out_df['threshold'].append(j)
        out_df['percentage'].append(n_up / n_all)
    return pd.DataFrame(out_df)


def f1_score(tp, fp, tn, fn):
    f1 = tp.copy()
    tp = tp['percentage']
    fp = fp['percentage']
    tn = tn['percentage']
    fn = fn['percentage']
    f1['percentage'] = tp / (tp + 0.5 * (fp + fn))
    return f1

within_df = get_threshold_array(pd.read_parquet('../within_df.parquet'))
across_df = get_threshold_array(pd.read_parquet('../across_df.parquet'))

tp = within_df
fp = across_df

tn = fp.copy()
tn['percentage'] = 1 - tn['percentage']

fn = within_df.copy()
fn['percentage'] = 1 - fn['percentage']

f1 = f1_score(tp, fp, tn, fn)

plt.plot(f1['percentage'], label='f1')
plt.plot(tp['percentage'], label='true positive')
plt.plot(fp['percentage'], label='false positive')
plt.legend()
plt.show()
