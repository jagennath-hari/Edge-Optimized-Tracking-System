#include <iostream>
#include <opencv2/opencv.hpp>
#include "BYTETracker.h"

int main() {
    // Initialize the BYTETracker
    BYTETracker tracker(30, 30);

    // Number of frames to simulate
    const int num_frames = 5;

    for (int frame = 1; frame <= num_frames; ++frame) {
        // Generate dummy detections for this frame
        std::vector<cv::Rect> detections = {
            cv::Rect(50 + frame * 5, 50 + frame * 2, 100, 100),
            cv::Rect(200 + frame * 3, 200 + frame * 3, 150, 150),
            cv::Rect(400 + frame * 2, 300 + frame * 4, 120, 180)
        };
        std::vector<int> classIds = {1, 2, 3};
        std::vector<float> scores = {0.9f, 0.85f, 0.7f};

        // Simulated empty frame (not used by BYTETracker but required by interface)
        cv::Mat frameImg;

        // Use the `track` method
        auto [trackBoxes, trackIds, trackScores] = tracker.track(frameImg, detections, classIds, scores);

        // Display the tracking results for this frame
        std::cout << "Frame " << frame << " Tracking Results:\n";
        for (size_t i = 0; i < trackBoxes.size(); ++i) {
            const auto& box = trackBoxes[i];
            std::cout << "Track ID: " << trackIds[i]
                      << ", Box: [" << box.x << ", " << box.y << ", " << box.width << ", " << box.height
                      << "], Score: " << trackScores[i] << "\n";
        }
        std::cout << "-------------------------------------\n";
    }

    return 0;
}
