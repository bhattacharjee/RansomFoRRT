#!/bin/bash
for i in ${i}*
do
    if [[ -d $i ]]; then
        echo Processing $i...
        ./convert_to_pandas.py -d $i
    fi
done

./combine_pandas.py
