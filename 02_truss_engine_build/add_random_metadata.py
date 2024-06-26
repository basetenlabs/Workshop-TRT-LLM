# Temporary hack
import random
import sys
import yaml

def prepend_model_metadata(filename):
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    random_integer = random.randint(100000, 999999)
    model_metadata = {
        "model_metadata": {
            "salt": random_integer
        }
    }
    data = {**model_metadata, **data}
    with open(filename, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

if __name__ == "__main__":
    prepend_model_metadata(sys.argv[1])