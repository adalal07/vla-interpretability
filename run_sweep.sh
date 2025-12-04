#!/bin/bash

python run_sweep.py --multirun\
    target=up,down,fast,slow \
    k=5,10,20 \
    alpha=2,4,6,8,10 \
    sweep_name=libero_valuevec_sweep
