#!/bin/bash

problem_dir="../examples/lotka-voltera/"
cd "$problem_dir"

nohup python 01_coda_serial.py > data/nohup.log 2>&1 &

nohup python main.py > tmp/nohup.log 2>&1 &