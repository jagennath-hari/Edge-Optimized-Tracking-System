#!/bin/bash

# Get the current directory
CURRENT_DIR=$(pwd)

# Set the container name
CONTAINER_NAME="cuda_tracker_container"

# Check if a dataset path is provided as an argument
DATASET_PATH=$1

# Function to start the container
start_container() {
  if [ -z "$DATASET_PATH" ]; then
    echo "Error: Dataset path is required to start a new container."
    echo "Usage: $0 <path_to_dataset>"
    exit 1
  fi

  echo "Starting a new container..."
  xhost + && docker run \
    --gpus all \
    -it \
    --privileged \
    --network=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --rm \
    -e DISPLAY=${DISPLAY} \
    --name "$CONTAINER_NAME" \
    -v $CURRENT_DIR/models/:/models \
    -v $CURRENT_DIR/weights/:/weights \
    -v $CURRENT_DIR/scripts/:/scripts \
    -v $DATASET_PATH:/dataset \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $CURRENT_DIR/tracker_system:/tracker_system \
    cuda_tracker:triton_base
}

# Function to attach to the running container
attach_container() {
  echo "Attaching to the running container..."
  docker exec -it "$CONTAINER_NAME" bash
}

# Check if the container is already running
if docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "$CONTAINER_NAME"; then
  attach_container
else
  # Check if the container exists but is stopped
  if docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "$CONTAINER_NAME"; then
    echo "Starting the stopped container..."
    docker start -ai "$CONTAINER_NAME"
  else
    # Start a new container
    start_container
  fi
fi

