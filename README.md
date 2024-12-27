# Edge-Optimized-Tracking-System
<div align="justify">
A high-performance multi-object tracking system utilizing a quantized YOLOv11 model deployed on the Triton Inference Server, integrated with a CUDA-accelerated particle filter for robust mutiple object tracking.
</div>

<div align="center">
    <img src="assets/result.gif" width="800" height="400" alt="Tracking System Result" />
    <p>Edge-Optimized Tracking System for the SportsMOT Dataset as an example.</p>
</div>

## 🏁 Dependencies
1) [NVIDIA Driver](https://www.nvidia.com/download/index.aspx)
2) [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
3) [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
4) [Docker Compose plugin](https://docs.docker.com/compose/install/linux/)

*Tested on Ubuntu 22.04 and with CUDA 12.1 using RTX 4090 GPU.*

## 🏋️ Pre trained weights for SportsMOT dataset
This trained network has only been trained on a single example dataset from the [SportsMOT dataset](https://github.com/MCG-NJU/SportsMOT). It was trained on the scoccer dataset specifically *v_gQNyhv8y0QY_c013*. [Sample Dataset on OneDrive from Authors](https://1drv.ms/u/s!AtjeLq7YnYGRgQRrmqGr4B-k-xsC?e=7PndU8)

[Pretrained Weights](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

### Training on custom dataset using YOLOv11
Training script [here](scripts/train.py).

### ONNX Conversion for YOLOv11
Conversion script [here](scripts/torch_to_onnx.py).

## 📐 System Design
<div align="center">
    <img src="assets/main_system_design.png" width="1500" height="200" alt="Main Sys Design" />
    <p>Overall System Design.</p>
</div>

<details>
    <summary>Perception Design</summary>
    This is divided into two components: one-time quantization and setting up the ensembled network for Triton Inference Server.

    <details>
        <summary>Quantization Framework</summary>
        <div align="center">
            <img src="assets/perception_quantization_design.png" width="1500" height="400" alt="Quantization Sys Design" />
            <p>Quantization framework.</p>
        </div>
        The exact command used for quantization in TensorRT can be found [here](weights/quantize_yolo.sh). For this example, FP16 was used.
    </details>

    <details>
        <summary>Inference for Triton Inference Server using ensembled model</summary>
        <div align="center">
            <img src="assets/perception_inference_design.png" width="1500" height="1000" alt="Perception Inference Sys Design" />
            <p>Inference framework.</p>
        </div>
        The entire source code for Perception is located [here](tracker_system/perception), and the ensembled model is [here](models).

        An example can be found [here](tracker_system/perception/examples/perception_dataset.cpp), which is a good starting point for making changes.
    </details>
</details>

<details>
    <summary>ByteTrack Design</summary>
    The [original authors' paper](https://arxiv.org/abs/2110.06864) was used. The [official repository](https://github.com/ifzhang/ByteTrack) gives a detailed explanation of the implementation.
</details>

<details>
    <summary>CUDA Particle Filter Design</summary>
    This implementation uses a complete GPU-accelerated Particle Filter with an additional Unscented Transform for the prediction step.

    <details>
        <summary>Structure of Array (SoA) for the states</summary>
        <div align="center">
            <img src="assets/particle_SoA.png" width="1500" height="1000" alt="Particle States Design" />
            <p>Particle States Structure of Array.</p>
        </div>
        The SoA is defined [here](tracker_system/filter/include/filter/particle_states.cuh) and [here](tracker_system/filter/src/particle_states.cu).
    </details>

    <details>
        <summary>CUDA Particle Filter with Unscented Transform</summary>
        <div align="center">
            <img src="assets/desgin_particle_filter_process.png" width="1500" height="1000" alt="Particle Filter Design" />
            <p>Particle Filter Process on the Device (GPU) with the Unscented Transform by propagating Sigma Points.</p>
        </div>
        The source code for the kernels is located [here](tracker_system/filter/include/filter/kernels.cuh) and [here](tracker_system/filter/src/kernels.cu).
    </details>
</details>
