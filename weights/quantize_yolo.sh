#!/bin/bash

export CUDA_MODULE_LOADING=LAZY && /usr/src/tensorrt/bin/trtexec --onnx=/weights/best.onnx --builderOptimizationLevel=5 --useSpinWait --useRuntime=full --useCudaGraph --precisionConstraints=obey --allowGPUFallback --tacticSources=+CUBLAS,+CUDNN,+JIT_CONVOLUTIONS,+CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS --saveEngine=/models/perception/1/model.plan  --workspace=5120 --layerOutputTypes=fp16 --layerPrecisions=fp16 --sparsity=enable --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
