# Makefile for setting up a Python virtual environment and installing dependencies

# Define the virtual environment directory
VENV_DIR = .env

# Define the path to the requirements file
REQUIREMENTS_FILE = requirements.txt

# Target to create a virtual environment
$(VENV_DIR):
	python3 -m venv $(VENV_DIR)

# Target to activate the virtual environment and install dependencies
install: $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -r $(REQUIREMENTS_FILE)

# Target to clean the virtual environment
clean:
	rm -rf $(VENV_DIR)

# Target to run the Python application
deploy_tiny_llama_on_baseten: install
	$(VENV_DIR)/bin/truss push ./tiny-llama-truss --publish --trusted

# Default target
all: install

.PHONY: install clean run all