#!/bin/bash

problem_dir="../examples/lotka-voltera/"
cd "$problem_dir"

nohup python 01_coda_serial.py > data/nohup.log 2>&1 &