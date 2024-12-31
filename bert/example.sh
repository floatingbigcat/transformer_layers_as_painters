#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

task='rte'
model='google-bert/bert-large-uncased'
start_layer=9 # start layer index
method='skip' # middle_repeat, skip, reverse, baseline, random, loop_parallel
repeat_time_or_seed=3 # repeat time for loop_parallel, seed number for random.
lr=2e-5 # 3e-2 if is_frozen == True
is_frozen='False'

python run_glue.py \
  --model_name_or_path $model \
  --task_name $task \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate $lr \
  --num_train_epochs 3 \
  --output_dir /tmp/$task/ \
  --method $method \
  --freeze_encoder $is_frozen \
  --repeat_time_or_seed 3 \
  --start_layer $start_layer \
  --fp16

# transformers==4.41.2, eval_accuracy=0.6354