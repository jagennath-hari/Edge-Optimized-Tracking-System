#pragma once

#include <opencv2/core.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <chrono> // For high-resolution clock

// Triton client and gRPC includes
#include "grpc_client.h"

// Include the IDetector interface
#include "detector_interface.hpp"

namespace perception 
{

    class TritonEngine : public IDetector 
    {
    public:
        // Constructor
        explicit TritonEngine();

        // Override the inference method from IDetector
        std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> infer(const cv::Mat& inputImage) override;

    private:
        // Triton Client
        std::unique_ptr<triton::client::InferenceServerGrpcClient> tritonClient_;

        // Convert FP16 to FP32 - static, as it doesn't depend on instance state
        static float fp16ToFloat_(uint16_t fp16);

        // Prepare Triton input tensor
        triton::client::InferInput* createTritonInput(const std::string& name, const cv::Mat& image, const std::string& datatype);

        // Extract outputs as a tuple - marked const to clarify it doesn't modify `result`
        std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> extractOutputs_(const triton::client::InferResult* result) const;
    };

} // namespace perception
