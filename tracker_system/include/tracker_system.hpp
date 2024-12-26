#pragma once

#include "detector_interface.hpp" // IDetector interface
#include "tracker_interface.hpp"  // ITracker interface
#include "filter_interface.hpp" // IFilter interface
#include <cmath>
#include <memory>
#include <vector>
#include <deque>
#include <tuple>
#include <iostream>

namespace tracking_system 
{

    class TrackerSystem 
    {
    public:
        /// @brief Constructor
        TrackerSystem(std::shared_ptr<IDetector> detector, std::shared_ptr<ITracker> tracker, std::shared_ptr<IFilter> filter);

        /// @brief Process a single video frame
        /// @param frame Input frame (cv::Mat)
        /// @return A vector of tracked objects (bounding boxes, IDs, scores, and particles)
        std::vector<std::tuple<std::vector<float>, int, float, std::vector<std::vector<float>>>> processFrame(const cv::Mat& frame);

    private:
        std::shared_ptr<IDetector> detector_; ///< Pointer to the detection module
        std::shared_ptr<ITracker> tracker_;  ///< Pointer to the tracking module
        std::shared_ptr<IFilter> filter_; ///< Pointer to the filter module

        // History of bounding boxes for each object (fixed length of 2)
        std::unordered_map<int, std::deque<cv::Rect>> history_;

        // Compute the quaternion using arc tan
        std::tuple<float, float, float, float> computeQuaternion_(float velocity_x, float velocity_y); 

        // Function to compute velocity
        std::pair<float, float> computeVelocity_(const cv::Rect& prev_bbox, const cv::Rect& curr_bbox);
    };

} // namespace tracking_system
