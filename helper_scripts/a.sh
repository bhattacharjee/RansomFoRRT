#!/bin/bash
# 3 4 5 6 7 8 9

export PATH=/usr/local/bin:$PATH

function wait_for_completion()
{
    NUM_PROCS=`ps -ef | grep iterate.py | grep -v plaintext_or |grep python| grep -v grep | grep -vw vim | grep -vw nvim | wc -l`
    while [[ $NUM_PROCS -ge 10 ]]; do
        NUM_PROCS=`ps -ef | grep iterate.py | grep -v plaintext_or |grep python| grep -v grep | grep -vw vim | grep -vw nvim | wc -l`
        sleep 5
    done
}
function wait_for_completion_all()
{
    NUM_PROCS=`ps -ef | grep iterate.py | grep -v plaintext_or |grep python| grep -v grep | grep -vw vim | grep -vw nvim | wc -l`
    while [[ $NUM_PROCS -ne 0 ]]; do
        NUM_PROCS=`ps -ef | grep iterate.py | grep -v plaintext_or |grep python| grep -v grep | grep -vw vim | grep -vw nvim | wc -l`
        sleep 5
    done
}

wait_for_completion_all

for j in *
do
    if [[ -d $j ]]; then
        ./iterate.py --directory $j &
        wait_for_completion
    fi
done

wait_for_completion_all
