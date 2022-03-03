#!/bin/bash
# 3 4 5 6 7 8 9

export PATH=/usr/local/bin:$PATH

function wait_for_completion()
{
    NUM_PROCS=55
    while [[ $NUM_PROCS -ne 0 ]]; do
        NUM_PROCS=`ps -ef | grep iterate.py |grep python| grep -v grep | grep -vw vim | grep -vw nvim | wc -l`
        sleep 5
    done
}
for i in 0
do
    for j in ${i}*
    do
        ./iterate.py --directory $j &
    done
done
wait_for_completion

for i in 1
do
    for j in ${i}*
    do
        ./iterate.py --directory $j &
    done
done
wait_for_completion

for i in 2
do
    for j in ${i}*
    do
        ./iterate.py --directory $j &
    done
done
wait_for_completion

for i in 3
do
    for j in ${i}*
    do
        ./iterate.py --directory $j &
    done
done
wait_for_completion

for i in 4
do
    for j in ${i}*
    do
        ./iterate.py --directory $j &
    done
done
wait_for_completion

for i in 5
do
    for j in ${i}*
    do
        ./iterate.py --directory $j &
    done
done
wait_for_completion

for i in 6
do
    for j in ${i}*
    do
        ./iterate.py --directory $j &
    done
done
wait_for_completion

for i in 7
do
    for j in ${i}*
    do
        ./iterate.py --directory $j &
    done
done
wait_for_completion

for i in 8
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
