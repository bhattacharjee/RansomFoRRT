#!/bin/bash
# 3 4 5 6 7 8 9

function wait_for_completion()
{
    NUM_PROCS=55
    while [[ $NUM_PROCS -ne 0 ]]; do
        NUM_PROCS=`ps -ef | grep iterate.py | grep -v grep | wc -l`
        sleep 5
    done
}
for i in 0 1 2
do
    for j in ${i}*
    do
        ./iterate.py --directory $j &
    done
done
wait_for_completion

for i in 3 4 5
do
    for j in ${i}*
    do
        ./iterate.py --directory $j &
    done
done
wait_for_completion

for i in 6 7 8
do
    for j in ${i}*
    do
        ./iterate.py --directory $j &
    done
done
wait_for_completion

for i in 9
do
    for j in ${i}*
    do
        ./iterate.py --directory $j &
    done
done
wait_for_completion
