#! /usr/bin/bash
(./compile.py -I -R 64 -C phhh1.mpc >output) && (./replicated-ring-party.x -I -p 0 phhh1 --batch-size 1000000 >output0 & (./replicated-ring-party.x -I -p 1 -v phhh1 --batch-size 1000000 > output1 2>&1) & ./replicated-ring-party.x -I -p 2 phhh1 --batch-size 1000000 > output2) && cat output1
