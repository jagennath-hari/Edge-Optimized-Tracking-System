#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>

/// @brief Interface for object tracking modules.
class ITracker 
{
public:
    /// @brief Virtual destructor for the interface.
    virtual ~ITracker() = default;

    /**
     * @brief Update the tracker with new detections and return tracking results.
     *
     * @param frame The current video frame.
     * @param detections A vector of bounding boxes for detected objects.
     * @param classIds A vector of class IDs corresponding to the detections.
     * @param scores A vector of confidence scores for the detections.
     * @return A tuple containing:
     *         - A vector of tracking bounding boxes (cv::Rect).
     *         - A vector of corresponding class IDs (int).
     *         - A vector of confidence scores (float).
     */
    virtual std::tuple<std::vector<cv::Rect>, std::vector<int>, std::vector<float>> track(const cv::Mat& frame, const std::vector<cv::Rect>& detections, const std::vector<int>& classIds, const std::vector<float>& scores) = 0;
};
