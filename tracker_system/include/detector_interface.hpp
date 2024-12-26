#pragma once

#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

/// @brief Interface for object detection modules.
class IDetector 
{
public:
    /// @brief Virtual destructor for the interface.
    virtual ~IDetector() = default;

    /**
    * @brief Perform inference on a given frame.
    * 
    * @param frame The input image (cv::Mat) to perform inference on.
    * @return A tuple containing:
    *         - A vector of cv::Rect representing bounding boxes.
    *         - A vector of floats representing confidence scores.
    *         - A vector of integers representing class IDs.
    */
    virtual std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> infer(const cv::Mat& frame) = 0;
};