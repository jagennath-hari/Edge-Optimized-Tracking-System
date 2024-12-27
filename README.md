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

[Pretrained Weights](https://drive.google.com/uc?export=download&id=13M0WVGBIsjVfTDMfZRp0fw7apFz1Fgn1)

Extract the file and place the ```best.pt``` and ```best.onnx``` in the ```weights``` folder.

### Training on custom dataset using YOLOv11
Training script [here](scripts/train.py).

### ONNX Conversion for YOLOv11
Conversion script [here](scripts/torch_to_onnx.py).

## 📐 System Design
<details>
<summary>Full pipeline</summary>

<div align="center">
    <img src="assets/main_system_design.png" width="1500" height="200" alt="Main Sys Design" />
    <p>Overall System Design.</p>
</div>

The overall system is divided into individual sub-systems, Perception, ByteTracker, and Particle Filter. Each of the sub-systems are explained below.

### Perception Design
This again is divided into two components which is the one time quantization, then the setting up the ensembled network for Triton Inference Server.

#### Quantization Framework
<div align="center">
    <img src="assets/perception_quantization_design.png" width="1500" height="400" alt="Quantization Sys Design" />
    <p>Quantization framework.</p>
</div>

The exact command used for quatization in TensoRT can be found [here](weights/quantize_yolo.sh), for this example FP16 was used.

#### Inference for Triton Inference Server using ensembled model
<div align="center">
    <img src="assets/perception_inference_design.png" width="1500" height="1000" alt="Perception Inference Sys Design" />
    <p>Inference framework.</p>
</div>

#### Full Pipeine
The entire source code for Perception is [here](tracker_system/perception) and the esembled model is located [here](models).

An example [here](tracker_system/perception/examples/perception_dataset.cpp) is a good starting point while making changes.


### ByteTrack Design
The [orginal authors paper](https://arxiv.org/abs/2110.06864) was used, the [Offical Reposiory](https://github.com/ifzhang/ByteTrack) gives a detailed explantion of the implementation.

### CUDA Particle Filter Design
This implementation uses a complete GPU accelerated Particle Filter with an additional Unscented Transform for the prediction step.

#### Structre of Array (SoA) for the states
We use a total of 10 states.

<div align="center">
    <img src="assets/particle_SoA.png" width="1500" height="1000" alt="Particle States Design" />
    <p>Particle States Structre of Array.</p>
</div>

The SoA is defined [here](tracker_system/filter/include/filter/particle_states.cuh) and [here](tracker_system/filter/src/particle_states.cu).

#### CUDA Particle Filter with Unscented Transform
<div align="center">
    <img src="assets/desgin_particle_filter_process.png" width="1500" height="1000" alt="Particle States Design" />
    <p>Particle Filter Process on the Device(GPU) with the Unscented Transform by propogating Sigma Points.</p>
</div>

The souce code for the kernels is located [here](tracker_system/filter/include/filter/kernels.cuh) and [here](tracker_system/filter/src/kernels.cu).
  
</details>

## 🏗️ Building the 🐳 Docker file
Start building the docker container.
```
bash build.sh
```

Compiling the code (One time process).
```
bash compile.sh
```
## ⌛️ Running on sample data
To run the composed container with Triton and the executable.
```
DATASET_PATH=/path/to/your/dataset bash run_and_exit.sh
```
The output video gets saved in the ```/tracker_system/result``` folder.

## 📖 Citation
If you found this code/work to be useful in your own research, please considering citing the following:
```bibtex
@software{Jocher_Ultralytics_YOLO_2023,
author = {Jocher, Glenn and Qiu, Jing and Chaurasia, Ayush},
license = {AGPL-3.0},
month = jan,
title = {{Ultralytics YOLO}},
url = {https://github.com/ultralytics/ultralytics},
version = {8.0.0},
year = {2023}
}
```
```bibtex
@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## 🪪 License
This software is released under BSD-3-Clause license. You can view a license summary [here](LICENSE). [Ultralytics](https://github.com/ultralytics/ultralytics) and [ByteTrack](https://github.com/ifzhang/ByteTrack) have their own licenses respectively.