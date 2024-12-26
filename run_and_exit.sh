#!/bin/bash

# Check if DATASET_PATH is provided and valid
if [ -z "$DATASET_PATH" ]; then
  echo "Error: DATASET_PATH environment variable is not set."
  echo "Usage: DATASET_PATH=/path/to/dataset ./run_and_exit.sh"
  exit 1
fi

if [ ! -d "$DATASET_PATH" ]; then
  echo "Error: The directory specified in DATASET_PATH does not exist: $DATASET_PATH"
  exit 1
fi

# Export DATASET_PATH to make it available to docker-compose
export DATASET_PATH

# Start docker-compose in detached mode
docker-compose up --exit-code-from cpp_tracker

# Wait for the cpp_tracker container to finish
docker wait cpp_tracker

# Bring down the entire docker-compose setup
docker-compose down

