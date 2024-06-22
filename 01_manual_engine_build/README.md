# Build TensortRT-LLM engine 

## Spin up a machine on Runpod

- Use RTX-4090 
- `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04` image
- 1 GPU is sufficient for this exercise
  - 2 GPUs if you want to try TensorParallelism
- Increase container disk to 200 GB in the template, we need more space
- Feel free to use spot to save cost
- ssh into the machine

## Verify that machine is set up well

```sh
nvidia-smi
```

You should see a box with 1 GPU with 24 GB of memory. You will also see cuda,
driver version and other things. If `nvidia-smi` doesn't work for some reason
then something is wrong and you should spin up the machine fresh.

## Install TensorRT-LLM

```sh
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs
pip3 install tensorrt_llm==0.11.0.dev2024061800 -U --pre --extra-index-url https://pypi.nvidia.com
```

This should take 5-10 mins

Verify that TensorRT-LLM is installed correctly:

```sh
python3 -c "import tensorrt_llm"
```

You should see something like:

```sh
[TensorRT-LLM] TensorRT-LLM version: 0.11.0.dev2024061800
```

## Clone TensorRT-LLM git repo to access llama example

```sh
# This should take a 3-5 mins
git clone https://github.com/NVIDIA/TensorRT-LLM
cd TensorRT-LLM

# Make sure were at a version compatible with the installed TensorRT-LLM library installed via pip above
git checkout 2a115dae84f13daaa54727534daa837c534eceb4

cd examples/llama
```

## Build Engine

Enable hf_transfer for faster download.

```sh
export HF_HUB_ENABLE_HF_TRANSFER=true
pip install hf_transfer

```

```sh
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir tllama

# Convert huggingface model a format that TensorRT-LLM works with
python3 convert_checkpoint.py --model_dir ./tllama/ \
                              --output_dir ./tllm_checkpoint_1gpu_fp16 \
                              --dtype float16

# Build Engine
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16 \
            --output_dir ./trt_engines/tllama/fp16/1-gpu \
            --gemm_plugin auto

# Check the size of engine created
du -h -d 1 ./trt_engines/tllama/fp16/1-gpu

python3 ../run.py --max_output_len=50 \
               --tokenizer_dir ./tllama/ \
               --engine_dir=./trt_engines/tllama/fp16/1-gpu
```

## Try FP8

```sh
## fp8

```sh
# Quantize HF tinyllama into FP8 and export trtllm checkpoint
python3 ../quantization/quantize.py --model_dir ./tllama \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir ./tllm_checkpoint_1gpu_fp8 \
                                   --calib_size 36

# Build trtllm engines from the trtllm checkpoint
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp8 \
             --output_dir ./trt_engines/tllama/fp8/1-gpu \
             --gemm_plugin auto

# Check the size of engine created
du -h -d 1 ./trt_engines/tllama/fp8/1-gpu

python3 ../run.py --max_output_len=50 \
               --tokenizer_dir ./tllama/ \
               --engine_dir=./trt_engines/tllama/fp8/1-gpu
```
