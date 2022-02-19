#!/usr/bin/env python3
import random
from datetime import datetime
import os
from functools import reduce
import pickle
import sys

random.seed(datetime.now())

filename = "https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/{}.zip"


try:
    random_choices = pickle.load(open("random_choices.p", "rb"))
except Exception:
    random_choices = random.choices(list(range(1000)), k=25)
    pickle.dump(random_choices, open("random_choices.p", "wb"))


def download_file(number):
    number = f"{number:03d}"
    fname = filename.format(number)
    print(f"Downloading {fname}... ")
    os.system(f"wget {fname}")


for filen in random_choices:
    download_file(filen)
