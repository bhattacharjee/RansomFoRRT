#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scipy, scipy.stats
import tqdm

PARQUET_FILENAME = "../xor_psds.parquet"

df = pd.read_parquet(PARQUET_FILENAME)

filenames = np.asarray(df['filename'].unique())

def get_psd(df, keylen):
    df = df[df['keylen'] == keylen]
    columns = [c for c in df.columns if c.startswith('psd')]
    return df[columns].to_numpy().flatten()

def compare_within_file(df, filename):
    out_arr = []
    df = df[df['filename'] == filename]
    keylengths = np.asarray(df['keylen'].unique())
    for i in range(len(keylengths) - 1):
        for j in range(i+1, len(keylengths)):
            p1 = get_psd(df, i)
            p2 = get_psd(df, j)
            pr = scipy.stats.pearsonr(p1, p2)
            out_arr.append((i, j, pr[0], pr[1]))
    return out_arr


def compare_within_files():
    dictdf = {
        'i': [],
        'j': [],
        'correlation': [],
        'confidence': []
    }
    for filename in tqdm.tqdm(filenames):
        for i, j, correlation, confidence in compare_within_file(df, filename):
            dictdf['i'].append(i)
            dictdf['j'].append(j)
            dictdf['correlation'].append(correlation)
            dictdf['confidence'].append(confidence)
    return pd.DataFrame(dictdf)

def compare_across_files():
    dictdf = {
        'filename1': [],
        'filename2': [],
        'i': [],
        'j': [],
        'correlation': [],
        'confidence': []
    }
    file_pairs = []
    for i in range(len(filenames)):
        for j in range(i+1, len(filenames)):
            filename1 = filenames[i]
            filename2 = filenames[j]
            file_pairs.append((filename1, filename2))
    for filename1, filename2 in tqdm.tqdm(file_pairs):
        df1 = df[df['filename'] == filename1]
        df2 = df[df['filename'] == filename2]
        keylengths = np.asarray(df1['keylen'].unique())
        for kl1 in keylengths:
            for kl2 in keylengths:
                p1 = get_psd(df1, kl1)
                p2 = get_psd(df2, kl2)
                correlation, confidence = scipy.stats.pearsonr(p1, p2)
                dictdf['filename1'].append(filename1)
                dictdf['filename2'].append(filename2)
                dictdf['i'].append(kl1)
                dictdf['j'].append(kl2)
                dictdf['correlation'].append(correlation)
                dictdf['confidence'].append(confidence)
    return pd.DataFrame(dictdf)



within_df = compare_within_files()
across_df = compare_across_files()
within_df.to_parquet('../within_df.parquet')
across_df.to_parquet('../across_df.parquet')
