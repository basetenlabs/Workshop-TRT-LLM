import argparse
import asyncio
import os
import time

from transformers import AutoTokenizer
from baseten_client import BasetenAsyncClient


BASETEN_API_KEY_ENV_VAR = "BASETEN_API_KEY"


# Define the async function to make a single request
async def make_request(client, data):
  async for response in client.predict(data):
    yield response


async def consume(async_iter):
  cnt = 0
  start_time = time.time()
  ttft = None
  async for _ in async_iter:
    if cnt == 0:
      ttft = time.time() - start_time
    cnt +=1
  return cnt, ttft


def parse_args():
  parser = argparse.ArgumentParser(description="Script to process input with specified arguments")

  parser.add_argument(
      "--model_base_url",
      type=str,
      required=True,
      help="Specify the output length"
  )
  parser.add_argument(
      "--hf_tokenizer",
      type=str,
      default="NousResearch/Llama-2-7b-hf",
      help="Specify the Hugging Face tokenizer"
  )
  parser.add_argument(
      "--concurrency",
      type=int,
      default=64,
      help="Specify the concurrency level"
  )
  parser.add_argument(
      "--input_len",
      type=int,
      default=10,
      help="Specify the input length"
  )
  parser.add_argument(
      "--output_len",
      type=int,
      default=1000,
      help="Specify the output length"
  )
  parser.add_argument(
      "--num_runs",
      type=int,
      default=2,
      help="number of runs"
  )

  return parser.parse_args()

async def run(model_base_url, input_len, output_len, concurrency, tokenizer):
  """
  Do a benchmark run.

  Sends requests in parallel and calculates various metrics such as time taken.

  Note that ttft here is worse that it would normally be. All requests are
  started at the same time, so all requests content on the context phase. In
  real world requests won't start at the same time. This can be fixed by adding
  a jitter to the start time of each request. But that's not implemented here to
  keep this code simple.
  """
  async with BasetenAsyncClient(
    url=model_base_url, # todo, remove mc-dev
    api_key=os.environ.get(BASETEN_API_KEY_ENV_VAR)
  ) as client:

    # Generate sample data
    input_ids = [i for i in range(input_len)]
    sample_data = [
      {
        "prompt": tokenizer.decode(input_ids),
        "max_tokens": output_len,
      }
      for i in range(concurrency)
    ]

    start_time = time.time()

    # Make concurrent requests
    tasks = [consume(make_request(client, data)) for data in sample_data]
    counts = await asyncio.gather(*tasks)
    total_output_tokens = sum([count[0] for count in counts])
    ttft = sum([count[1] for count in counts]) / len(counts)
    # print(counts)

    end_time = time.time()

    # Calculate total time taken
    total_time = end_time - start_time
    print("\tTotal time taken:", total_time, "seconds")
    print("\tOutput Tps:", (total_output_tokens / total_time))
    print("\tTotal Tps:", (concurrency * input_len + total_output_tokens) / total_time)
    print("\tTTFT:", ttft * 1000, "ms")

async def main():

  args = parse_args()
  print(args)
  tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)
  print("Warmup run:")
  await run(args.model_base_url, 1, 1, 10, tokenizer)

  print("\nBenchmark runs:")
  for i in range(args.num_runs):
    print(f"Run {i+1}:")
    await run(
      args.model_base_url,
      args.input_len,
      args.output_len,
      args.concurrency,
      tokenizer,
    )


if __name__ == "__main__":
    asyncio.run(main())
