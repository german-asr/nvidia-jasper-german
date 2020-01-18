# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/bin/bash
echo "Container nvidia build = " $NVIDIA_BUILD_ID


MODEL_CONFIG=$1
DATA_DIR="/datasets"
CREATE_LOGFILE="true"
CUDNN_BENCHMARK="false"
PRECISION="fp32"
NUM_STEPS="-1"
SEED=0
BATCH_SIZE=64

for dataset in test_tuda test_common_voice dev_tuda dev_common_voice; do
    for checkpoint in /results/*.pt; do
        name=$(basename $checkpoint .pt)
        result_dir="/results/${name}"
        modeloutput_file="${result_dir}/${dataset}.logits"

        mkdir -p $result_dir


        if [ "$CREATE_LOGFILE" = "true" ] ; then
            export GBS=$(expr $BATCH_SIZE)
            printf -v TAG "jasper_inference_${dataset}_%s_gbs%d" "$PRECISION" $GBS
            DATESTAMP=`date +'%y%m%d%H%M%S'`
            LOGFILE="${result_dir}/${TAG}.${DATESTAMP}.log"
            printf "Logs written to %s\n" "$LOGFILE"
        fi

        PREC=""
        if [ "$PRECISION" = "fp16" ] ; then
            PREC="--fp16"
        elif [ "$PRECISION" = "fp32" ] ; then
            PREC=""
        else
            echo "Unknown <precision> argument"
            exit -2
        fi

        PRED=""
        OUTPUT=" --logits_save_to $modeloutput_file"


        if [ "$CUDNN_BENCHMARK" = "true" ]; then
            CUDNN_BENCHMARK=" --cudnn_benchmark"
        else
            CUDNN_BENCHMARK=""
        fi

        STEPS=""
        if [ "$NUM_STEPS" -gt 0 ] ; then
            STEPS=" --steps $NUM_STEPS"
        fi

        CMD=" python inference.py "
        CMD+=" --batch_size $BATCH_SIZE "
        CMD+=" --dataset_dir $DATA_DIR "
        CMD+=" --val_manifest $DATA_DIR/full_jasperized/${dataset}.json"
        CMD+=" --model_toml $MODEL_CONFIG  "
        CMD+=" --seed $SEED "
        CMD+=" --ckpt $checkpoint "
        CMD+=" $CUDNN_BENCHMARK"
        CMD+=" $PRED "
        CMD+=" $OUTPUT "
        CMD+=" $PREC "
        CMD+=" $STEPS "


        set -x
        if [ -z "$LOGFILE" ] ; then
           $CMD
        else
           (
             $CMD
           ) |& tee "$LOGFILE"
        fi
        set +x
        echo "MODELOUTPUT_FILE: ${modeloutput_file}"
        echo "PREDICTION_FILE: ${PREDICTION_FILE}"

    done
done
