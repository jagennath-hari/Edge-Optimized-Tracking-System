#include "tracker_system.hpp"

namespace tracking_system 
{

    TrackerSystem::TrackerSystem(
        std::shared_ptr<IDetector> detector, 
        std::shared_ptr<ITracker> tracker, 
        std::shared_ptr<IFilter> filter
    ) : detector_(std::move(detector)), 
        tracker_(std::move(tracker)), 
        filter_(std::move(filter))
    {
        if (!(this->detector_)) throw std::invalid_argument("Detector cannot be null");
        if (!(this->tracker_)) throw std::invalid_argument("Tracker cannot be null");
        if (!(this->filter_)) throw std::invalid_argument("Filter cannot be null");
    }

    // Function to calculate quaternion from velocity and normalize it
    std::tuple<float, float, float, float> TrackerSystem::computeQuaternion_(float velocity_x, float velocity_y) 
    {
        // Calculate angle of motion
        float angle = std::atan2(velocity_y, velocity_x); // Angle in radians

        // Compute quaternion components
        float qw = std::cos(angle / 2.0f);
        float qx = 0.0f; // Assuming 2D motion, qx and qy are 0
        float qy = 0.0f;
        float qz = std::sin(angle / 2.0f);

        // Normalize the quaternion
        float norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        qw /= norm;
        qx /= norm;
        qy /= norm;
        qz /= norm;

        return {qw, qx, qy, qz};
    }

    // Function to compute velocity
    std::pair<float, float> TrackerSystem::computeVelocity_(const cv::Rect& prev_bbox, const cv::Rect& curr_bbox) 
    {
        cv::Point2f prev_position(prev_bbox.x + prev_bbox.width / 2.0f, prev_bbox.y + prev_bbox.height / 2.0f);
        cv::Point2f curr_position(curr_bbox.x + curr_bbox.width / 2.0f, curr_bbox.y + curr_bbox.height / 2.0f);

        float velocity_x = curr_position.x - prev_position.x;
        float velocity_y = curr_position.y - prev_position.y;
        return {velocity_x, velocity_y};
    }

    std::vector<std::tuple<std::vector<float>, int, float, std::vector<std::vector<float>>>> TrackerSystem::processFrame(const cv::Mat& frame) 
    {
        // Step 1: Perform object detection
        auto [detections, scores, classIds] = this->detector_->infer(frame);

        // Step 2: Pass detections to the tracker
        auto [trackedBoxes, trackedIds, trackedScores] = this->tracker_->track(frame, detections, classIds, scores);

        std::vector<std::tuple<std::vector<float>, int, float, std::vector<std::vector<float>>>> results;

        for (size_t i = 0; i < trackedBoxes.size(); ++i) 
        {
            int object_id = trackedIds[i];
            cv::Rect bbox = trackedBoxes[i];
            float score = trackedScores[i];

            // Add the current bounding box to the history
            this->history_[object_id].push_back(bbox);

            // Ensure the history size doesn't exceed 2
            if (this->history_[object_id].size() > 2) 
            {
                this->history_[object_id].pop_front();
            }

            // Compute velocity and quaternion if history has 2 elements
            float velocity_x = 0.0f, velocity_y = 0.0f;
            std::tuple<float, float, float, float> quaternion = {1.0f, 0.0f, 0.0f, 0.0f};
            if (this->history_[object_id].size() == 2) 
            {
                const auto& prev_bbox = this->history_[object_id][0];
                const auto& curr_bbox = this->history_[object_id][1];
                std::tie(velocity_x, velocity_y) = this->computeVelocity_(prev_bbox, curr_bbox);
                std::tuple<float, float, float, float> quaternion = this->computeQuaternion_(velocity_x, velocity_y);
            }

            // Perform particle filter processing
            cv::Point2f position(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
            if (!this->filter_->isTrackingObject(object_id)) 
            {
                auto [qw, qx, qy, qz] = quaternion;
                this->filter_->initialize(object_id, position.x, position.y, velocity_x, velocity_y,
                                          bbox.width, bbox.height, qw, qx, qy, qz);
            } 
            else 
            {
                // Predict step for the filter
                this->filter_->predict(object_id);

                auto [qw, qx, qy, qz] = quaternion;
                this->filter_->updateObject(object_id, position.x, position.y, velocity_x, velocity_y,
                                            bbox.width, bbox.height, qw, qx, qy, qz);
            }

            // Get the filtered state
            std::vector<float> filtered_state = this->filter_->getObjectState(object_id);

            // Get all particles
            std::vector<std::vector<float>> particles = this->filter_->getObjectParticles(object_id);

            // Add the result to the output
            results.emplace_back(filtered_state, object_id, score, particles);
        }

        return results;
    }

} // namespace tracking_system
