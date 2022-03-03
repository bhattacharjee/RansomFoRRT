#!/usr/bin/env python3
import scipy
import os
import glob
import json
import logging
import argparse
import numpy as np
from scipy import signal
import tqdm
import dit

from scipy.stats import entropy

FORCE_RECALCULATION = False
FORCE_FOURIER_RECALCULATION = False

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



#-------------------------------------------------------------------------------


def get_dit_entropies(filename, bytearray, doublearray, metadata):
    """
        Use the dit library to get shannon and Renyi entropies
    """

    def get_dit_entropy(bytearray, name):
        counts = None
        values = None
        probabilities = None
        d = None

        if "advanced" not in metadata:
            metadata["advanced"] = {}

        # Renyi 0 through 10
        for i in range(11):
            keyname = f"dit.renyi.{name}.{i}"
            if keyname not in metadata["advanced"]:
                if counts is None or values is None or d is None:
                    counts, values = np.histogram(bytearray,bins=256,range=(0, 256))
                    values = values[:-1]
                    probabilities = counts / bytearray.shape[0]
                    d = dit.ScalarDistribution(values, probabilities)
                metadata["advanced"][keyname] = \
                    dit.other.renyi_entropy(d, order=i, rvs=None, rv_mode='names')

        # Renyi infinity (most probable event)
        keyname = f"dit.renyi.{name}.inf"
        if keyname not in metadata["advanced"]:
            if counts is None or values is None or d is None:
                counts, values = np.histogram(bytearray,bins=256,range=(0, 256))
                values = values[:-1]
                probabilities = counts / bytearray.shape[0]
                d = dit.ScalarDistribution(values, probabilities)
            metadata["advanced"][keyname] = \
                dit.other.renyi_entropy(d, order=np.inf, rvs=None, rv_mode='names')

        keyname = f"dit.shanon.{name}"
        if keyname not in metadata["advanced"]:
            if counts is None or values is None or d is None:
                counts, values = np.histogram(bytearray,bins=256,range=(0, 256))
                values = values[:-1]
                probabilities = counts / bytearray.shape[0]
                d = dit.ScalarDistribution(values, probabilities)
            metadata["advanced"][keyname] = dit.shannon.entropy(d)

    get_dit_entropy(bytearray[:128], "begin")
    get_dit_entropy(bytearray[-128:], "tail")
    get_dit_entropy(bytearray, "full")

    return metadata

                


#-------------------------------------------------------------------------------



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





#-------------------------------------------------------------------------------





def get_filesize(filename, bytearray, doublearray, metadata):
    """
        Get the file size
    """
    if not "baseline" in metadata:
        metadata["basleine"] = {}

    if not "filesize" in metadata["baseline"] or FORCE_RECALCULATION:
        metadata["baseline"]["filesize"] = len(bytearray)

    return metadata





#-------------------------------------------------------------------------------




def get_chisquare(filename, bytearray, doublearray, metadata):
    """
        Get chisquare estimate
    """
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
    



#-------------------------------------------------------------------------------



def get_moments(filename, bytearray, doublearray, metadata):
    """
        Get statistical moments (an array of them)
    """

    def get_moment(doublearray, keyname, n):
        if keyname not in metadata["advanced"] or FORCE_RECALCULATION:
            moment = scipy.stats.moment(doublearray, moment=n)
            metadata["advanced"][keyname] = moment

    if "advanced" not in metadata:
        metadata["advanced"] = {}

    for i in range(2, 16):
        get_moment(doublearray, f"moment_full.{i}", i)
        get_moment(doublearray[:128], f"moment_begin.{i}", i)
        get_moment(doublearray[-128:], f"moment_end.{i}", i)

    return metadata



#-------------------------------------------------------------------------------




def get_autocorrelation(filename, bytearray, doublearray, metadata):
    if not "baseline" in metadata:
        metadata["baseline"] = {}
    
    def autocorrelation(doublearray, metadata, name):
        name = f"autocorrelation_{name}"
        if name not in metadata["baseline"] or FORCE_RECALCULATION:
            metadata["baseline"][name] = \
                np.corrcoef(doublearray[:-1], doublearray[1:])[1, 0]
        return metadata

    metadata = autocorrelation(doublearray, metadata, "full")
    metadata = autocorrelation(doublearray[:128], metadata, "begin")
    metadata = autocorrelation(doublearray[-128:], metadata, "end")

    return metadata



#-------------------------------------------------------------------------------



def get_kurtosis(filename, bytearray, doublearray, metadata):
    """
        Get the kurtosis of a file
    """
    def kurtosis(doublearray, metadata, name):
        name = f"kurtosis_{name}"
        if name not in metadata["advanced"] or FORCE_RECALCULATION:
            metadata["advanced"][name] = \
                scipy.stats.kurtosis(doublearray)
        return metadata

    if "advanced" not in metadata:
        metadata["advanced"] = {}

    metadata = kurtosis(doublearray, metadata, "full")
    metadata = kurtosis(doublearray[-128:], metadata, "end")
    metadata = kurtosis(doublearray[:128], metadata, "begin")
    return metadata

    

#-------------------------------------------------------------------------------



def get_skew(filename, bytearray, doublearray, metadata):
    """
        Get the skew of the data
    """
    def skew(doublearray, metadata, name):
        name = f"skew_{name}"
        if name not in metadata["advanced"] or FORCE_RECALCULATION:
            metadata["advanced"][name] = scipy.stats.skew(doublearray)
        return metadata

    metadata = skew(doublearray, metadata, "full")
    metadata = skew(doublearray[:128], metadata, "begin")
    metadata = skew(doublearray[-128:], metadata, "end")
    
    return metadata



#-------------------------------------------------------------------------------


def get_fourier_psd(filename, byte1_array, doublearray, metadata):
    """
        Get the fourier spectrum of the data
    """
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
                indexes_to_ignore = [i-2, i-1, i, i+1, i+2]
        if len(indexes_to_ignore) > 0:
            f = [f[i] for i in range(len(f)) if i not in indexes_to_ignore]
            p = [p[i] for i in range(len(p)) if i not in indexes_to_ignore]
        return f, p


    def get_welch(sequence):
        """
            Get the welch power spectrum for the sequence of bytes
        """
        # Mostly it returns the same frequencies, but not always
        # We will ignore if frequencies are different
        freq, psd = signal.welch(sequence, return_onesided = False)
        freq, psd = get_sorted_spectrum(freq, psd)
        return np.array(freq), np.array(psd)

    def get_stats(nparray):
        """
            Get some stats for the fourier transform power spectrum:
            mean, autocorrelation, and standard deviation
        """
        f_autocorr = np.corrcoef(nparray[:-1], nparray[1:])[1, 0]
        f_mean = np.mean(nparray)
        f_std = np.std(nparray)
        return f_mean, f_std, f_autocorr

    def process_buffer(nparray, name):
        """
            Now that we have the fourier array, add it to the json
        """
        f, p = get_welch(nparray)

        # Add some stats about the fourier power spectrum
        f_autocorr, f_mean, f_std = get_stats(p)
        metadata["fourier"][f"stat.{name}.autocorr"] = f_autocorr
        metadata["fourier"][f"stat.{name}.mean"] = f_mean
        metadata["fourier"][f"stat.{name}.std"] = f_std
        for n in range(2, 16):
            moment = scipy.stats.moment(p, n)
            metadata["fourier"][f"stat.{name}.moment.{n}"] = moment

        # Finally add the full fourier spectrum in case we need it
        for n, power in enumerate(p):
            metadata["fourier"][f"value.{name}.{n}"] = float(power)

    def expand_buffer(inbuffer, nbytes):
        """
            Buffer should be as large as nperseg * nbytes (256 * nbytes)
            where nbytes is the number of bytes that were read together
            for each entry of the numpy array, in our case 4 and 1
        """
        buffer = b''
        while len(buffer) <= 256 * nbytes:
            buffer += inbuffer
        return buffer

    padding = [b'', b'000', b'00', b'0']
    if "fourier" not in metadata \
        or FORCE_FOURIER_RECALCULATION or FORCE_RECALCULATION:
        with open(filename, "rb") as fp:
            byte1_array = bytearray(fp.read())
            byte1_array = expand_buffer(byte1_array, 1)
            byte4_array = expand_buffer(byte1_array, 4)

            nparray = np.frombuffer(byte1_array, dtype=np.ubyte)
            doublearray_1byte = nparray.astype(np.double)

            paddedarray = byte4_array + padding[len(byte4_array) % 4]
            nparray = np.frombuffer(paddedarray, dtype=np.intc)
            doublearray_4byte = nparray.astype(np.double)

            metadata["fourier"] = {}
            process_buffer(doublearray_1byte, "1byte")
            process_buffer(doublearray_4byte, "4byte")

    return metadata

#-------------------------------------------------------------------------------


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################


def get_metadata_filename(filename):
    """
        Given a file path, return the full path to its metadata file
    """
    dirname = os.path.dirname(filename)
    filename = os.path.basename(filename)
    filename = filename.split("/")[-1]
    return f"{dirname}{os.path.sep}__metadata_{filename}.json"




#-------------------------------------------------------------------------------



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
        get_chisquare,
        get_autocorrelation,
        get_filesize,

        # Advanced metrics

        get_kurtosis,
        get_skew,
        get_moments,
        get_dit_entropies,

        # Fourier spectrum
        get_fourier_psd
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
                metadata = process_fn(filename, nparray, doublearray, metadata)
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
