#!/usr/bin/env bash
set -e
python src/train.py --train-csv data/train.csv --output-dir models
