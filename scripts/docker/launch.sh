#!/bin/bash
source /etc/bash.bashrc

SCRIPT_DIR=$(cd $(dirname $0); pwd)
JASPER_REPO=${JASPER_REPO:-"${SCRIPT_DIR}/../.."}

# Launch TRT JASPER container.

mkdir -p $2
mkdir -p $3

DATA_DIR=$(realpath $1)
CHECKPOINT_DIR=$(realpath $2)
RESULT_DIR=$(realpath $3)
PROGRAM_PATH=${PROGRAM_PATH}

MOUNTS=""
if [ ! -z "$DATA_DIR" ];
then
    MOUNTS="$MOUNTS -v $DATA_DIR:/datasets "
fi

if [ ! -z "$CHECKPOINT_DIR" ];
then
    MOUNTS="$MOUNTS -v $CHECKPOINT_DIR:/checkpoints "
fi

if [ ! -z "$RESULT_DIR" ];
then
    MOUNTS="$MOUNTS -v $RESULT_DIR:/results "
fi

echo $MOUNTS
nvidia-docker run -it --rm \
  --runtime=nvidia \
  --shm-size=4g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  ${MOUNTS} \
  -v ${JASPER_REPO}:/jasper \
  ${EXTRA_JASPER_ENV} \
  jasper:latest bash $PROGRAM_PATH
