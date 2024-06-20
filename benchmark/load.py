import aiohttp
import argparse
import asyncio
import os
import time

from transformers import AutoTokenizer


BASETEN_API_KEY_ENV_VAR = "BASETEN_API_KEY"


# Define the async function to make a single request
async def make_request(client, data):
  async for response in client.predict(data):
    yield response


async def consume(async_iter):
    async for item in async_iter:
        pass


class BasetenAsyncClient:
  def __init__(self, base_url, api_key, timeout=120):
    self.base_url = base_url
    self.api_key = api_key
    self.timeout = timeout
    self.session = None

  async def __aenter__(self):
    self.session = aiohttp.ClientSession(
      headers={
        "Authorization": f"Api-Key {self.api_key}",
        "Content-Type": "application/json"
      }
    )
    return self

  async def __aexit__(self, exc_type, exc_value, traceback):
    await self.session.close()

  async def predict(self, model_input):
    url = f"{self.base_url}/production/predict"
    async with self.session.post(url, json=model_input, timeout=self.timeout) as response:
      response.raise_for_status()  # Raise an exception for HTTP errors
      async for chunk in response.content.iter_any():
        yield chunk.decode('utf-8')


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

  return parser.parse_args()

async def main():

  args = parse_args()
  print(args)
  tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)

  # todo: 
  # 1. do multiple runs and average
  # 2. calc ttft
  # 3. calc output tps
  async with BasetenAsyncClient(
    base_url=args.model_base_url, # todo, remove mc-dev
    api_key=os.environ.get(BASETEN_API_KEY_ENV_VAR)
  ) as client:

    # Generate sample data
    input_ids = [i for i in range(args.input_len)]
    sample_data = [
      {
        "prompt": tokenizer.decode(input_ids),
        "max_tokens": args.output_len,
      }
      for _ in range(args.concurrency)
    ]

    start_time = time.time()

    # Make concurrent requests
    tasks = [consume(make_request(client, data)) for data in sample_data]
    results = await asyncio.gather(*tasks)

  end_time = time.time()

  # Calculate total time taken
  total_time = end_time - start_time
  print("Total time taken:", total_time, "seconds")
  print("Tps:", (args.concurrency * args.output_len) / total_time)


if __name__ == "__main__":
    asyncio.run(main())
