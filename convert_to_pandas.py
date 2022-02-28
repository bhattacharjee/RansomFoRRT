#!/usr/bin/env python3


import glob
import json
import argparse
import tqdm
import os
import sys
import logging
import pandas as pd

def get_json(filename):
    if os.path.isfile(filename) and "metadata" in filename and "json" in filename:
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error processing {filename}")
            print(e)
            return None
    else:
        return None


PARALLEL_JOBS = 24

#-------------------------------------------------------------------------------

def flatten_json(injson):
    outjson = {}
    for outercol in ["extended", "baseline", "advanced"]:
        for k in injson[outercol]:
            outjson[f"{outercol}.{k}"] = injson[outercol][k]
    return outjson


def iterate_files(base_dir):
    """
        Iterate over a directory, and for every file that is not a metadata
        file, call process_single_file.

        Metadata files have the following format:
        __metadata_origfname.origextn.json
    """
    savedir = None
    accumulator = []
    try:
        savedir = os.getcwd()
        os.chdir(base_dir)
        #filenames = glob.glob("./**", recursive=True)
        filenames = glob.glob("__metadata*json", recursive=True)
        for i in tqdm.tqdm(range(0, len(filenames), PARALLEL_JOBS)):
            for item in map(get_json, filenames[i: i + PARALLEL_JOBS]):
                if item is not None:
                    accumulator.append(item)
            if i == 5:
                break
        accumulator = list(map(flatten_json, accumulator))
        TEMPFILENAME = "/tmp/temp.json"
        if os.path.exists(TEMPFILENAME):
            os.remove(TEMPFILENAME)
        with open(TEMPFILENAME, "w") as f:
            json.dump(accumulator, f)
        df = pd.read_json(TEMPFILENAME, orient='records')
        os.chdir(savedir)
        df.to_parquet(f"{base_dir}.parquet.gzip", compression='gzip')
    except Exception as e:
        logging.error(f"Exception in iteration {e}")
        raise e
    finally:
        if savedir is not None:
            os.chdir(savedir)




#-------------------------------------------------------------------------------



def main():
    parser = argparse.ArgumentParser(\
        description="Iterate json files and produce a pandas dataframe parquet")
    parser.add_argument("--directory", "-d", type=str, required=True)
    args = parser.parse_args()
    iterate_files(args.directory)



#-------------------------------------------------------------------------------



if "__main__" == __name__:
    main()
