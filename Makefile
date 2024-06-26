MODEL_BASE_URL ?= 
INPUT_LEN ?= 1000
OUTPUT_LEN ?= 100
VENV_DIR = .env
REQUIREMENTS_FILE = requirements.txt
CONCURRENCY ?= 64
NUM_RUNS ?= 2

$(VENV_DIR):
	python3 -m venv $(VENV_DIR)

install: $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -r $(REQUIREMENTS_FILE)

clean:
	rm -rf $(VENV_DIR)

deploy_tiny_llama_on_baseten: install
	# Hack to disable caching, so full model build flow is exercised for inspection
	$(VENV_DIR)/bin/python ./02_truss_engine_build/add_random_metadata.py ./02_truss_engine_build/tiny-llama-truss/config.yaml

	$(VENV_DIR)/bin/truss push ./02_truss_engine_build/tiny-llama-truss --publish --trusted

benchmark:
	$(VENV_DIR)/bin/python3 03_benchmark/load.py \
	  --model_base_url $(MODEL_BASE_URL) \
		--input_len $(INPUT_LEN) \
		--output_len $(OUTPUT_LEN) \
		--concurrency $(CONCURRENCY) \
		--num_runs $(NUM_RUNS)

default: install

.PHONY: install clean all benchmark deploy_tiny_llama_on_baseten