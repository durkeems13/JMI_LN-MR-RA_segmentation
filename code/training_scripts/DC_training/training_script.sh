#!/bin/bash

export TRAIN_RUN=$1
export TRAIN_LOGDIR=../../train_logs/$TRAIN_RUN

if [ ! -d $TRAIN_LOGDIR ]; then
    mkdir $TRAIN_LOGDIR 
fi

##FINE TUNE--change lines 80 and 108 in config.py
#python3 train.py --load=../../train_logs/MP_Tcell_detection/checkpoint --logdir=../../train_logs/MP_Tcell_detection_resume1

##Train from scratch
python3 train.py --logdir=$TRAIN_LOGDIR

