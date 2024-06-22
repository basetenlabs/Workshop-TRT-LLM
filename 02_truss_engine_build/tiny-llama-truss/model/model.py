"""
The `Model` class is an interface between the ML model that you're packaging and
the model server that you're running it on.

When trt_llm.build section is provided in config.yaml, like in this case, this
class is automatically generated and below is just a placeholder.
"""


class Model:
    def __init__(self, **kwargs):
        pass

    def load(self):
        pass

    def predict(self, model_input):
        pass
