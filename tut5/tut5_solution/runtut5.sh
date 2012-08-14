#!/bin/sh

export PGPLOT_DIR=/usr/lib64/pgplot

make clean

make

python2.7 tut5_gencoeff.py -n 32768 -t 8 -b 1 -d float 

bin/tut5 -b 1 -n 32768 -p
