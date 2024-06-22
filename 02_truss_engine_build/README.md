## Build and deploy a TensorRT engine with Truss

Let's try building an engine for the same model automatically.

We're beta testing a new feature of Truss -- you're the first people outside of the company to see this!

## Sign up for Baseten

To run the Truss build, we'll use Baseten. When you create an account, you'll get $30 of free credits. This tutorial will consume less than a dollar of credits.

Sign up here: [https://app.baseten.co/signup/](https://app.baseten.co/signup/)

If you experience any issues signing up for Baseten, such as getting placed in an approval queue, let Philip know and he'll get you access.

## Edit the Truss config

Your TensorRT-LLM build is configured entirely in `config.yaml`:

```yaml
model_name: tiny-llama-a10g
python_version: py310
resources:
  accelerator: A10G
  use_gpu: True
system_packages:
  - python3.10-venv
requirements:
  - sentencepiece
trt_llm:
  build:
    max_input_len: 2000
    max_output_len: 2000
    max_batch_size: 64
    max_beam_width: 1
    base_model: llama
    quantization_type: weights_int8
    checkpoint_repository:
      repo: TinyLlama/TinyLlama-1.1B-Chat-v1.0
      source: HF
```

See the [Truss codebase](https://github.com/basetenlabs/truss/blob/59b9fed62ee5805c7a68e97c63532c463ae15a88/truss/config/trt_llm.py) for more options.

## Deploy the Truss to commence engine build

Via the Truss CLI:

```sh
pip install truss
truss push ./tiny-llama-truss --publish --trusted
```

or via make

```sh
make deploy_tiny_llama_on_baseten
```

While the engine build and model deployment run, head over to your Baseten workspace to see logs.