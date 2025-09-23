#!/bin/bash

PROJECT_DIR=/inspire/hdd/global_user/liao-liao/workspace/siiRL-VLA
export PYTHONPATH="$PROJECT_DIR:/root/LIBERO/:/inspire/hdd/global_user/liao-liao/projects/vjepa2:$PYTHONPATH"
cd $PROJECT_DIR
bash examples/vla_rl.sh 2>&1 | tee siirl_vla_train.log
