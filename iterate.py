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

FORCE_RECALCULATION = False

class Metric:
    def __init__(self):
        self.__instance = None

    def get_instance():
        return None

    @property
    def instance(self):
        return get_instance()



def get_extension(filename, bytearray, doublearray, metadata):
    """
        Return the extension of a file
    """
    if not "extended" in metadata:
        metadata["extended"] = {}
    if "extension" in metadata["extended"] and not FORCE_RECALCULATION:
        return metadata
    try:
        extension = os.path.splitext(filename)[-1]
        metadata["extended"]["extension"] = extension
    except:
        metadata["extended"]["extension"] = "null"
    return metadata

def get_basename(filename):
    """
        Return the basename of a file
    """
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

    if "baseline" not in metadata.keys():
        metadata["baseline"] = {}
    nparr = doublearray

    if "head_shannon_entropy" not in metadata["baseline"] or FORCE_RECALCULATION:
        # Get entropy of the head bytes
        try:
            metadata["baseline"]["head_shannon_entropy"] = \
                get_entropy_internal(nparr[:256])
        except Exception as e:
            metadata["baseline"]["head_shannon_entropy"] = -1.0
            raise e

    # get entropy of the tail bytes
    if "tail_shannon_entropy" not in metadata["baseline"] or FORCE_RECALCULATION:
        try:
            metadata["baseline"]["tail_shannon_entropy"] = \
                get_entropy_internal(nparr[-257:])
        except Exception as e:
            metadata["baseline"]["tail_shannon_entropy"] = -1.0
            raise e

    # get entropy of all bytes
    if "shannon_entropy" not in metadata["baseline"] or FORCE_RECALCULATION:
        try:
            metadata["baseline"]["shannon_entropy"] = get_entropy_internal(nparr)
        except Exception as e:
            metadata["baseline"]["shannon_entropy"] = -1.0
            raise e
    return metadata

def get_montecarlo_pi(filename, bytearray, doublearray, metadata):
    """
        Fill the montecarlo estimation of pi
    """
    trials = 0
    counts = 0

    if not "baseline" in metadata:
        metadata["baseline"] = {}

    if "montecarlo_pi" in metadata["baseline"] and not FORCE_RECALCULATION:
        return metadata

    doublearray = doublearray - np.min(doublearray)
    doublearray = doublearray / np.max(doublearray)
    for i in range(0, len(doublearray), 2):
        try:
            trials += 1
            x1 = doublearray[i]
            x2 = doublearray[i + 1]
            if (x1 * x1 + x2 * x2 <= 1):
                counts += 1
        except:
            break

    try:
        mcp = 4.0 * counts / trials
    except Exception:
        mcp = 0.0

    metadata["baseline"]["montecarlo_pi"] = mcp
    return metadata


def get_chisquare(filename, bytearray, doublearray, metadata):
    if not "baseline" in metadata:
        metadata["baseline"] = {}

    def get_chisquare_int(doublearray, metadata, name):
        if name not in metadata["baseline"] or FORCE_RECALCULATION:
            metadata["baseline"][name] = \
                scipy.stats.chisquare(doublearray).statistic
        return metadata

    metadata = get_chisquare_int(doublearray, metadata, "chisquare_full")
    metadata = get_chisquare_int(doublearray[:128], metadata, "chisquare_begin")
    metadata = get_chisquare_int(doublearray[-128:], metadata, "chisquare_end")
    return metadata
    

def get_metadata_filename(filename):
    """
        Given a file path, return the full path to its metadata file
    """
    dirname = os.path.dirname(filename)
    filename = os.path.basename(filename)
    filename = filename.split("/")[-1]
    return f"{dirname}{os.path.sep}__metadata_{filename}.json"

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
        get_entropy,
        get_montecarlo_pi,
        get_chisquare
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

def process_file_if_required(filename):
    """
        If a file is a regular file but not a metadata file, only then
        process the file to extract information about it and store the same
        in the metadata file.
    """
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
            list(map(process_file_if_required, filenames[i: i + PARALLEL_JOBS]))
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
