#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

start_layer=14 # start layer index
model=meta-llama/Llama-2-7b-hf #mistralai/Mistral-7B-v0.1
method='skip' # middle_repeat, skip, reverse, baseline, random, loop_parallel
repeat_time_or_seed=3 # repeat time if method is loop_parallel, seed if method is random
tasks='lambada_openai' # arc_challenge,hellaswag,winogrande,gsm8k

lm_eval --model hf \
    --model_args pretrained=$model \
    --tasks $tasks \
    --batch_size 8 \
    --method $method \
    --start_layer $start_layer \
    --repeat_time_or_seed $repeat_time_or_seed

# |    Tasks     |Version|Filter|n-shot|  Metric  |Value |   |Stderr|
# |--------------|------:|------|-----:|----------|-----:|---|-----:|
# |lambada_openai|      1|none  |     0|perplexity|3.5034|±  |0.0701|
# |              |       |none  |     0|acc       |0.7270|±  |0.0062|
