#!/bin/bash
#BSUB -J QNET_TR
#BSUB -o quaternet_train_%J.out
#BSUB -q gpuv100
#BSUB -W 02:30
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"

# Variables
SUBJECT=
SRC_DIR=Humo_project
WEIGHT_FILE=weights_long_term.bin
DATA_DIR=$SRC_DIR/datasets
WORK_DIR=quaternet_clone_${LSB_JOBID}
TRAIN_SCRIPT=train_long_term.py

# Commands
if [ "${PWD##*/}" == "$SRC_DIR" ]; then
	# If we are currently within the source dir, then move out one layer
	echo In source directory. Moving to parent directory...
	cd ..
	echo done.
fi
if [ -d "$WORK_DIR" ]; then
	# Abort if working directory already exists - maybe something interesting is in there
        echo Working directory \'$WORK_DIR\' already exists. Aborting...
        exit
fi
echo Loading modules...
module load cuda/10.0
module load python3
echo done.
if [ -f bin/activate ]; then
	echo Activating virtualenv...
	source bin/activate
	echo done.
else
	echo Virtualenv not found. Aborting...
	exit
fi

git clone $SRC_DIR $WORK_DIR           # Clone source code
echo Symlinking to data...
mkdir -p $WORK_DIR/datasets            # Make dir for data
ln $DATA_DIR/* -t $WORK_DIR/datasets   # Make links to data
#ln $SRC_DIR/$WEIGHT_FILE -t $WORK_DIR  # Make link to weights
echo done.

echo Running training script \($TRAIN_SCRIPT\)...
cd $WORK_DIR                           # Change to working directory
python3 -u $TRAIN_SCRIPT $SUBJECT         # Run training script
echo done.
echo Exiting...
