#!/usr/bin/env python3
import sys
import scipy
import os
import glob
import matplotlib.pyplot as plt
import json
import logging
import argparse
import numpy as np
import pandas as pd
from scipy import signal
import sklearn
from sklearn.preprocessing import MinMaxScaler
import tqdm
import shutil

from scipy.stats import entropy

FORCE_RECALCULATION = False
FORCE_FOURIER_RECALCULATION = False
window_fn = 'boxcar'
TEMP_FILENAME = "temp.txt"

def get_fourier_psd(filename):
    """
        Get the fourier spectrum of the data
    """
    infilename = filename
    def get_sorted_spectrum(f, p):
        """
            Sort the power spectrum by frequency
        """
        arr = sorted([(f[i], p[i]) for i in range(len(f))])
        f = [a[0] for a in arr]
        p = [a[1] for a in arr]

        # Ignore stuff around the zero frequency as energies are usually very low
        # there and tends to give wrong results
        indexes_to_ignore = []
        for i, n in enumerate(f):
            if n == 0.0:
                pass
                #indexes_to_ignore = [i-2, i-1, i, i+1, i+2]
        if len(indexes_to_ignore) > 0:
            f = [f[i] for i in range(len(f)) if i not in indexes_to_ignore]
            p = [p[i] for i in range(len(p)) if i not in indexes_to_ignore]
        return f, p


    def get_welch(sequence):
        """
            Get the welch power spectrum for the sequence of bytes
        """
        def hpass(nb):
            nonlocal infilename
            onb = nb
            nb = nb.astype(dtype=np.float32)
            samples = nb.shape[0]
            # nb = nb / 255.0
            return nb

        # Mostly it returns the same frequencies, but not always
        # We will ignore if frequencies are different
        freq, psd = signal.welch(hpass(sequence), window=window_fn, nperseg=4096, return_onesided=False)
        freq, psd = get_sorted_spectrum(freq, psd)
        freq, psd = np.array(freq), np.array(psd)
        psd[psd < 5000] = 0.0
        return freq, psd

    with open(filename, 'rb') as f:
        b = bytearray(f.read())
        nb = np.frombuffer(b, dtype=np.ubyte)
        a, b = get_welch(nb)
        return a, b

def xor_file(filename: str, outfilename: str, key: bytearray):
    with open(filename, 'rb') as f:
        b = bytearray(f.read())
        indtext, indkey = 0, 0
        for indtext in range(len(b)):
            b[indtext] = b[indtext] ^ key[indkey]
            indkey = (indkey + 1) % len(key)

        with open(outfilename, 'wb') as wf:
            wf.write(b)
            wf.close()



def xor_file_multiple(filename: str, outfilename: str, key: bytearray):
    if os.path.exists(filename):
        psd_dict = {}
        if os.path.exists(outfilename): os.remove(outfilename)
        psd_dict[0] = get_fourier_psd(filename)[1]
        for i in range(1, len(key) + 1):
            xor_file(filename, TEMP_FILENAME, key[:i])
            psd_dict[i] = get_fourier_psd(TEMP_FILENAME)[1]
            try:
                os.remove(TEMP_FILENAME)
            except:
                pass
        return psd_dict


def get_psds_for_all_files(directory: str, key):
    dictdf = {}
    if not directory.endswith('/'): directory += '/'
    directory += '*'
    for filename in tqdm.tqdm(glob.glob(directory)):
        if filename.endswith('.py') or filename.endswith('.txt'):
            continue
        x = xor_file_multiple(filename, outfilename=TEMP_FILENAME, key=key)
        if "filename" not in dictdf.keys(): dictdf["filename"] = []
        if "keylen" not in dictdf.keys(): dictdf["keylen"] = []
        for i in range(len(x[0])):
            if f'psd_{i}' not in dictdf.keys():
                dictdf[f'psd_{i}'] = []
        for i, xx in x.items():
            dictdf['filename'].append(filename)
            dictdf['keylen'].append(i)
            for v in range(len(xx)):
                dictdf[f'psd_{v}'].append(xx[v])
    return pd.DataFrame(dictdf)

if '__main__' == __name__:
    if 2 <= len(sys.argv):
        window_fn = sys.argv[1]
        print('---------------------------------------------')
        print('WINDOW = ', window_fn)
    dictdf = get_psds_for_all_files(directory='.', key=b'abcdef')
    dictdf.to_parquet(path="../xor_psds.parquet")
        
