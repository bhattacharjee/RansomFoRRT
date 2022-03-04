#!/usr/bin/env python3
import os
import glob
import json
import logging
import argparse
import tqdm
import logging

from scipy.stats import entropy

FORCE_RECALCULATION = False
FORCE_FOURIER_RECALCULATION = False

CUTOFF_SIZE = 1024 + 512

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
        metadata["extended"]["base_filename"] = os.path.basename(filename)
    except:
        metadata["extended"]["extension"] = "null"
        metadata["extended"]["base_filename"] = os.path.basename(filename)
    return metadata



#-------------------------------------------------------------------------------



def get_basename(filename):
    """
        Return the basename of a file
    """
    try:
        basename = os.path.basename(filename)
    except:
        basename = "null"
    return basename

#-------------------------------------------------------------------------------

def process_single_file(filename):
    st = os.stat(filename)
    length = st.st_size
    if length < CUTOFF_SIZE:
        temp_filename = f"{filename}.__temp.tmp"
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
        out_buffer = b''
        with open(filename, "rb") as f:
            in_buffer = bytearray(f.read())
            while len(out_buffer) < CUTOFF_SIZE:
                out_buffer += in_buffer
            with open(temp_filename, "wb") as f:
                f.write(out_buffer)
        if os.path.exists(temp_filename):
            os.unlink(filename)
            os.rename(temp_filename, filename)
            logging.debug(f"Overwrote {filename}")


#-------------------------------------------------------------------------------



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




#-------------------------------------------------------------------------------


PARALLEL_JOBS = 4



#-------------------------------------------------------------------------------


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
        filenames = glob.glob("./**", recursive=True)
        for i in tqdm.tqdm(range(0, len(filenames), PARALLEL_JOBS)):
            list(map(process_file_if_required, filenames[i: i + PARALLEL_JOBS]))
    except Exception as e:
        logging.error(f"Exception in iteration {e}")
        raise e
    finally:
        if savedir is not None:
            os.chdir(savedir)




#-------------------------------------------------------------------------------



def main():
    parser = argparse.ArgumentParser(\
        description="Iterate files and add statistics")
    parser.add_argument("--directory", "-d", type=str, required=True)
    args = parser.parse_args()
    iterate_files(args.directory)



#-------------------------------------------------------------------------------



if "__main__" == __name__:
    main()
