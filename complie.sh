#!/bin/bash

# Get the current directory
CURRENT_DIR=$(pwd)

# Set the container name
CONTAINER_NAME="cuda_tracker_container"

# Function to start the container, compile the code, and run the script
start_container_and_compile() {
  echo "Starting a new container, compiling code, and running quantize script..."
  
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
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $CURRENT_DIR/tracker_system:/tracker_system \
    cuda_tracker:triton_base \
    bash -c "cd /tracker_system && \
             cmake -DCMAKE_BUILD_TYPE=Release -DPERCEPTION_BUILD_EXAMPLES=ON -DBYTETRACKER_BUILD_EXAMPLES=ON -DFILTER_BUILD_EXAMPLES=ON -S . -B build && \
             cmake --build build -j\$(nproc) && \
             bash /weights/quantize_yolo.sh"
}

# Just run the process — no stopping, attaching, or checking for running containers
start_container_and_compile
