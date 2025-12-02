#!/bin/bash

python run_sweep.py --multirun\
    target=up,down \
    k=3,4 \
    alpha=0.05 \
    top_k_tokens=5 \
    eval.n_episodes=2 \
    sweep_name=libero_valuevec_sweep
