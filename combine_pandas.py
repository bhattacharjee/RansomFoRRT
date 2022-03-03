#!/usr/bin/env python3
import os
import glob
import pandas as pd
import tqdm

def combine_csv_files(root_dir):
    PARQUET_FILE = "./combined.parquet.gz"
    CSV_FILE = "./combined.csv.gz"
    cwd = os.getcwd()
    try:
        os.chdir(root_dir)

        if os.path.exists(CSV_FILE): os.unlink(CSV_FILE)
        if os.path.exists(PARQUET_FILE): os.unlink(PARQUET_FILE)

        all_dfs = []
        files = list(glob.glob("./*.csv"))
        print("Reading CSV files...")
        for f in tqdm.tqdm(files):
            all_dfs.append(pd.read_csv(f))
        combined_df = pd.concat(all_dfs).copy()

        print("Writing combined csv file...")
        combined_df.to_csv(\
            CSV_FILE,\
            compression={'method': 'gzip', 'compresslevel': 9})
        print("Done.")

        print("Writing combined parquet file")
        combined_df.to_parquet(PARQUET_FILE, compression='gzip')
        print("Done")

        print("Deleting individual CSV files...")
        for f in tqdm.tqdm(files):
            os.unlink(f)

    except Exception as e:
        print("Failure occurred")
        raise e
    finally:
        os.chdir(cwd)
        
combine_csv_files(".")
