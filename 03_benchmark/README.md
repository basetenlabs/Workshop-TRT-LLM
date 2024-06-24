# Benchmarking

1. Make sure the model is deployed and running and that concurrency target is more than max
   concurrency you want to try,
   1. e.g. 32 for the example below
   2. To update this click on `Configure autoscaling` on your model page and set
      `Concurrency target`
2. In Baseten UI for your deployed model, click on call model and obtain the
   model predict url
   1. It looks like
      `https://model-<model_version_id>.api,baseten.co/production/predict`
3. Log into Baseten, generate and api key and copy it
4. export BASETEN_API_KEY=<your api key>
5. Run benchmark `make benchmark MODEL_BASE_URL=<model_predict_url_from_above> CONCURRENCY=32 OUTPUT_LEN=1000 INPUT_LEN=1000`

Feel free to modify various settings such as INPUT_LEN, OUTPUT_LEN and
CONCURRENCY to see how it impacts ttft and tps.

Please note that concurrency 1 and concurrency == max_batch_size are special
cases and worth trying.

1. Concurrency 1 provides best ttft and perceived tps (per request served tps)
2. Concurrency == max_batch_size provides max tps

## Exercise

See how far you can increase throughput by doing the following:

1. Increasing max batch size in
   `02_truss_engine_build/tiny-llama-truss/config.yaml`
2. Deploying
3. Benchmarking

## Modifying the benchmarking script

The [benchmarking script](03_benchmark/load.py) is a simple python script on
purpose. Feel free to study it and modify it. e.g. Instead of getting metrics
per run, try to average them over runs.