#!/usr/bin/env python3
import random
from datetime import datetime
import os
from functools import reduce
import pickle
import sys

random.seed(datetime.now())

filenamezz = "https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/{}.zip"


def download_file(number):
    print()
    print()
    print('-' * 80)
    number = f"{number:03d}"
    fname = filenamezz.format(number)
    print(f"Downloading {fname}... ")
    os.system(f"wget {fname}")

for i in range(25):
    filename = "null"
    num = -1
    while True:
        num = random.randint(0, 1000)
        filename = f"{num:03d}.zip"
        if not os.path.exists(filename):
            break
    download_file(num)
        
