#!/usr/bin/env python3
import scipy
import os
import glob
import json
import logging
import argparse
import numpy as np
import tqdm

from scipy.stats import entropy

def get_extension(filename, bytearray, doublearray, metadata):
    try:
        extension = os.path.splitext(filename)[-1]
        metadata["extension"] = extension
    except:
        metadata["extension"] = "null"
    return metadata

def get_basename(filename):
    try:
        basename = os.path.basename(filename)
    except:
        basename = "null"
    return basename

def get_entropy(filename, bytearray, doublearray, metadata):
    """
        Fill three types of entropies
        1. First 256 bytes
        2. Last 256 bytes
        3. Full file
    """
    def get_entropy_internal(nparr):
        value, counts = np.unique(nparr, return_counts=True)
        return scipy.stats.entropy(counts, base=2)
    nparr = doublearray
    try:
        metadata["head_shannon_entropy"] = get_entropy_internal(nparr[:256])
    except Exception as e:
        metadata["head_shannon_entropy"] = -1.0
        raise e
    try:
        metadata["tail_shannon_entropy"] = get_entropy_internal(nparr[-257:])
    except Exception as e:
        metadata["tail_shannon_entropy"] = -1.0
        raise e
    try:
        metadata["shannon_entropy"] = get_entropy_internal(nparr)
    except Exception as e:
        metadata["shannon_entropy"] = -1.0
        raise e
    return metadata

def get_metadata_filename(filename):
    dirname = filename.split("/")[:-1]
    dirname = "/".join(dirname)
    filename = filename.split("/")[-1]
    return f"{dirname}/__metadata_{filename}.json"

def process_single_file(filename):
    """
        1. Open the metadata file if it exists and populate the dictionary
        2. Open the file, and read the contents
            a. Copy the contents to a numpy character array
            b. Copy the contents into a numpy double array
        3. Iterate through all processing functions and call them
           one by one.
        4. Finally write the metadata into the disk
    """
    process_functions = [
        get_extension,
        get_entropy
    ]
    
    metadata = dict()
    nparray = None
    metadata_filename = get_metadata_filename(filename)
    try:
        f = open(f"{metadata_filename}", "r")
        metadata = json.load(f)
        f.close()
    except:
        metadata = dict()
    try:
        with open(filename, "rb") as f:
            nparray = bytearray(f.read())
            nparray = np.frombuffer(nparray, dtype=np.ubyte)
            doublearray = nparray.astype(np.double)
            for process_fn in process_functions:
                metadata = process_fn(filename, bytearray, doublearray, metadata)
                assert(metadata is not None and isinstance(metadata, dict))
    except Exception as e:
        logging.error(f"Exception in process_single_file {filename}")
        print(f"Exception in process_single_file {filename}")
        raise e
    else:
        try:
            with open(f"{metadata_filename}", "w") as f:
                json.dump(metadata, f)
        except Exception as e:
            logging.error(f"Error writing metadata to file for {filename} {e}")

def process_if_required(filename):
    if not get_basename(filename).startswith("__metadata") \
        and not get_basename(filename).endswith(".json") \
        and os.path.isfile(filename):
        process_single_file(filename)

PARALLEL_JOBS = 128
def iterate_files(base_dir):
    """
        Iterate over a directory, and for every file that is not a metadata
        file, call process_single_file.

        Metadata files have the following format:
        __metadata_origfname.origextn.json
    """
    savedir = None
    try:
        savedir = os.getcwd()
        os.chdir(base_dir)
        filenames = glob.glob("**", recursive=True)
        for i in tqdm.tqdm(range(0, len(filenames), PARALLEL_JOBS)):
            list(map(process_if_required, filenames[i: i + PARALLEL_JOBS]))
    except Exception as e:
        logging.error(f"Exception in iteration {e}")
        raise e
    finally:
        if savedir is not None:
            os.chdir(savedir)

def main():
    parser = argparse.ArgumentParser(\
        description="Iterate files and add statistics")
    parser.add_argument("--directory", "-d", type=str, required=True)
    args = parser.parse_args()
    iterate_files(args.directory)

if "__main__" == __name__:
    main()
