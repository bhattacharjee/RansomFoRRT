#!/bin/bash
for i in 0 1 2 3 4 5 6 7 8 9
do
    for j in ${i}*
    do
        echo Processing $j...
        ./convert_to_pandas.py -d $j
    done
done