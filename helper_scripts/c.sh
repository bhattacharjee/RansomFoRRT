#!/bin/bash
for i in 1 2 3 4 5 6 7 8
do
    for j in ${i}*
    do
        /Users/phantom/bin/encrypt_python_wrapper.sh $j \; &
    done
done
