#!/bin/bash

for i in *
do
    if [[ -d $i ]]; then
        find $i -name "*metadata*json" -exec rm -f {} \; &
    fi
done
