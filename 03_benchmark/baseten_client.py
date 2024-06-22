import aiohttp


class BasetenAsyncClient:
  def __init__(self, url, api_key, timeout=120):
    self.url = url
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
    url = f"{self.url}"
    async with self.session.post(url, json=model_input, timeout=self.timeout) as response:
      response.raise_for_status()  # Raise an exception for HTTP errors
      async for chunk in response.content.iter_any():
        yield chunk.decode('utf-8')
