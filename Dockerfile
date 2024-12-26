# Use Triton Inference Server base image
FROM nvcr.io/nvidia/tritonserver:23.06-py3

# Set the working directory
WORKDIR /cuda_tracker

# Default entrypoint to start Triton

ENV CUDA_HOME /usr/local/cuda
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Triton client
RUN pip3 install tritonclient[all]

# Install OpenCV
RUN apt-get update && apt-get install -y libgl1
RUN pip3 install opencv-contrib-python

# Add Triton Python Backend path to PYTHONPATH
ENV PYTHONPATH="/opt/tritonserver/backends/python"

# Install torch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Fix for hwloc
# Add the required library path to LD_LIBRARY_PATH
RUN apt-get update && apt-get install --reinstall -y \
  libmpich-dev \
  hwloc-nox libmpich12 mpich

RUN apt-get install libsm6 libice6 libglib2.0-0 -y

# Add the required library paths to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

# Install dependencies for building OpenCV with GStreamer and FFmpeg
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtk-3-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    libopenexr-dev \
    python3-dev \
    python3-numpy \
    libgstreamer1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libgtk-3-dev libatlas-base-dev libopenblas-dev liblapack-dev \
    libxvidcore-dev libx264-dev libswscale-dev libgphoto2-dev \
    libeigen3-dev libhdf5-dev libprotobuf-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Add more dependencies if needed
RUN apt-get update && apt-get install -y \
    libsm6 libice6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone OpenCV and OpenCV Contrib repositories
WORKDIR /opt
RUN apt-get update && apt-get install -y ninja-build

RUN git clone --branch 4.x https://github.com/opencv/opencv.git && \
    git clone --branch 4.x https://github.com/opencv/opencv_contrib.git && \
    mkdir -p opencv/build && cd opencv/build && \
    cmake -G Ninja \
          -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D OPENCV_ENABLE_NONFREE=ON \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_DOCS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_opencv_world=OFF \
          -D WITH_CUDA=ON \
          -D CUDA_FAST_MATH=ON \
          -D WITH_CUBLAS=ON \
          -D WITH_CUDNN=ON \
          -D WITH_OPENGL=ON \
          -D WITH_V4L=ON \
          -D WITH_TBB=ON \
          -D ENABLE_PRECOMPILED_HEADERS=OFF \
          -D CUDA_ARCH_BIN="7.5 8.0 8.6 8.9 9.0" \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          .. && \
    ninja && \
    ninja install && \
    ldconfig && \
    rm -rf /opt/opencv /opt/opencv_contrib

# Verify OpenCV installation
RUN pkg-config --modversion opencv4

RUN apt-get install -y \
    curl \
    libcurl4-openssl-dev \
    rapidjson-dev \
    zlib1g-dev

# Clone the Triton client repository
RUN git clone -b main --recurse-submodules https://github.com/triton-inference-server/client.git /opt/triton-client

# Build Triton C++ client
RUN mkdir -p /opt/triton-client/build && \
    cd /opt/triton-client/build && \
    cmake -DCMAKE_INSTALL_PREFIX=`pwd`/install -DTRITON_ENABLE_CC_HTTP=OFF -DTRITON_ENABLE_CC_GRPC=ON -DTRITON_ENABLE_PERF_ANALYZER=OFF -DTRITON_ENABLE_PYTHON_HTTP=OFF -DTRITON_ENABLE_PYTHON_GRPC=OFF -DTRITON_ENABLE_JAVA_HTTP=OFF -DTRITON_ENABLE_GPU=ON -DTRITON_ENABLE_EXAMPLES=OFF -DTRITON_ENABLE_TESTS=OFF -DTRITON_COMMON_REPO_TAG=main -DTRITON_THIRD_PARTY_REPO_TAG=main -DTRITON_CORE_REPO_TAG=main -DTRITON_BACKEND_REPO_TAG=main -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_EXTENSIONS=OFF .. && \
    make cc-clients -j$(nproc)
    
    
WORKDIR /tracker_system
