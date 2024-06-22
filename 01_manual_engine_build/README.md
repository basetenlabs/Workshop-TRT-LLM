# Build TensorRT-LLM engine 

Building a TensorRT-LLM engine requires a GPU. For this step, we'll use RTX 4090s on RunPod.

The fastest way to get everyone their own GPU is for you to sign up for your own account on RunPod. There's a ten dollar minimum credit buy on RunPod — if you're unable to expense this, let Philip know and he can reimburse the ten dollars after the workshop.

If you're unable to access RunPod for any reason, let Philip know and as a fallback he will spin up a GPU for you and get you web console access.

## Set up RunPod

1. [Sign up for an account](https://www.runpod.io/console/signup).
2. Purchase 10 dollars of credits (minimum purchase, completing the workshop will cost far less)
3. Create a public SSH key and paste it under SSH Public Keys in your [Account Settings](https://www.runpod.io/console/user/settings).

To create your keypair, open terminal and run:

```
ssh-keygen -t ed25519
```

When asked for a path, create a new key at `.ssh/runpod`

To get the public key, run:

```
cat .ssh/runpod.pub
```

Paste the entire value into the SSH Public Keys field in your [Account Settings](https://www.runpod.io/console/user/settings).

## Spin up a 4090

Let's get a GPU!

- Go to the Pods page and click [Deploy](https://www.runpod.io/console/deploy).
- Select RTX 4090 from the list
- Stick with 1 GPU, unless you want to try Tensor Parallelism, in which case you should set it to 2 GPUs
- Use the "Change Template" button to set the template to `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
- Use the "Edit Template" to increase the container disk to 200 GB to make space for installing dependencies
- Deploy!

Once the Pod is deployed, SSH into it locally or use the web console to access a shell through your browser.

## Verify that machine is set up well

```sh
nvidia-smi
```

You should see a box with 1 GPU with 24 GB of memory. You will also see cuda,
driver version and other things. If `nvidia-smi` doesn't work for some reason
then something is wrong and you should spin up the machine fresh.

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.08              Driver Version: 545.23.08    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4090        On  | 00000000:C1:00.0 Off |                  Off |
|  0%   27C    P8              11W / 450W |      3MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

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
