#!/bin/bash
set -x


PARQUET_FILENAME="${1}.parquet.gz"
CSV_FILENAME="${1}.csv.gz"
mv combined.csv.gz $CSV_FILENAME
mv combined.parquet.gz $PARQUET_FILENAME

cp $CSV_FILENAME /Volumes/Expansion/MSCDATA/processed_save/csv
cp $PARQUET_FILENAME /Volumes/Expansion/MSCDATA/processed_save/parquet
cp $CSV_FILENAME /Users/phantom/mscwork/processed_save/csv
cp $PARQUET_FILENAME /Users/phantom/mscwork/processed_save/parquet


