#!/bin/bash
for i in 1 2 3 4 5 6 7 8
do
    for j in ${i}*
    do
        find $j -exec /Users/phantom/bin/enc_aes_256.sh {} \; &
    done
done
