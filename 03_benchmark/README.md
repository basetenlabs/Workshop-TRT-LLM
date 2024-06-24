# Benchmark your TensorRT-LLM engine

Let's see how much performance we've unlocked.


## Getting set up

Watch a 1-minute video walkthrough (p.s. I already revoked the API key, don't try any funny business):

https://github.com/basetenlabs/Workshop-TRT-LLM/assets/98474633/4f9933ff-dd55-4749-93bc-163c1d3bc3ea

Here are the steps to get set up to run the script:

1. Make sure the model is deployed and running and that concurrency target is more than max
   concurrency you want to try e.g. 32 for the example below
   1. To update this click on `Configure autoscaling` on your model page and set
      `Concurrency target`
2. In Baseten UI for your deployed model, click on call model and obtain the
   model predict url
   1. It looks like
      `https://model-<model_id>.api,baseten.co/production/predict`
3. Log into Baseten, generate and api key and copy it
4. export `BASETEN_API_KEY=<your api key>`

## Running the benchmark script

```sh
make benchmark MODEL_BASE_URL=<model_predict_url_from_above> CONCURRENCY=32 OUTPUT_LEN=1000 INPUT_LEN=1000
```

Feel free to modify various settings such as `INPUT_LEN`, `OUTPUT_LEN` and
`CONCURRENCY` to see how it impacts TTFT and TPS.

Special cases worth trying:

1. `Concurrency == 1` provides best TTFT and perceived TPS (per request served)
2. `Concurrency == max_batch_size` provides the best TPS

## Benchmarking different configurations

See how far you can increase throughput by doing the following:

1. Increasing max batch size in
   `02_truss_engine_build/tiny-llama-truss/config.yaml`
2. Deploying
3. Benchmarking

## Modifying the benchmarking script

The [benchmarking script](/03_benchmark/load.py) is a simple python script on
purpose. Feel free to study it and modify it. e.g. Instead of getting metrics
per run, try to average them over runs.
