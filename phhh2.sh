#! /usr/bin/bash
(./compile.py -I -R 64  phhh2.mpc >output) && (./replicated-ring-party.x -I -p 0 phhh2 >output0 & (./replicated-ring-party.x -I -p 1 -v phhh2 > output1 2>&1) & ./replicated-ring-party.x -I -p 2 phhh2 > output2) && cat output1
