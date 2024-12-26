#include <iostream>
#include <chrono>
#include <filesystem>
#include <vector>
#include "filter/particle_filter.hpp" // Include ParticleFilter (IFilter implementation)
#include "perception/engine.hpp" // Include Perception (IDetector implementation)
#include "tracker_system.hpp" // Include the TrackerSystem class
#include "BYTETracker.h" // Include BYTETracker (ITracker implementation)
#include <opencv2/core.hpp> // Use core features only
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp> // For Rodrigues
#include <opencv2/videoio.hpp> // For VideoWriter

namespace fs = std::filesystem;

// Function to convert a quaternion to a rotation matrix
cv::Mat quaternionToRotationMatrix(float qw, float qx, float qy, float qz) 
{
    cv::Mat rotation_matrix = (cv::Mat_<double>(3, 3) << 
        1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
        2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
        2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy));
    return rotation_matrix;
}

int main(int argc, char* argv[]) {
    // Check if the user provided a dataset directory
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_directory>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string datasetDirectory = argv[1];

    try {
        // Verify the directory exists
        if (!fs::exists(datasetDirectory) || !fs::is_directory(datasetDirectory)) {
            throw std::runtime_error("Invalid directory: " + datasetDirectory);
        }

        // Collect and sort image files
        std::vector<std::string> imageFiles;
        for (const fs::directory_entry& entry : fs::directory_iterator(datasetDirectory)) {
            if (entry.is_regular_file()) {
                const std::string extension = entry.path().extension().string();
                if (extension == ".jpg" || extension == ".png" || extension == ".jpeg") {
                    imageFiles.push_back(entry.path().string());
                }
            }
        }

        std::sort(imageFiles.begin(), imageFiles.end());

        if (imageFiles.empty()) {
            throw std::runtime_error("No valid image files found in directory: " + datasetDirectory);
        }

        // Create detector, tracker, and filter instances
        std::shared_ptr<IDetector> detector = std::make_shared<perception::TritonEngine>();
        std::shared_ptr<ITracker> tracker = std::make_shared<BYTETracker>(30, 30);
        std::shared_ptr<IFilter> filter = std::make_shared<particle_filter::ParticleFilter>(
            100, 8.0f, 8.0f, 0.25f, 0.25f, 5.0f, 5.0f, 0.1f, 0.0f, 0.0f, 0.1f, 25
        );

        // Create the TrackerSystem
        tracking_system::TrackerSystem system(detector, tracker, filter);

        // Get the path to the first image
        const std::string& firstImagePath = imageFiles[0];

        // Load the first image
        cv::Mat firstImage = cv::imread(firstImagePath);

        if (firstImage.empty()) {
            std::cerr << "Failed to load the first image: " << firstImagePath << std::endl;
            return 1;
        }

        // Create a VideoWriter object
        // Generate filename with current date and time
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
        std::string filename = ss.str() + ".avi"; 
        // Specify the desired output path
        std::string output_path = "/tracker_system/result/"; // Replace with your desired path

        // Create full output file path
        std::string full_filename = output_path + filename;

        cv::VideoWriter videoWriter(full_filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, firstImage.size()); 

        int frame_count = 1;
        // Process each image
        for (const std::string& imageFile : imageFiles) {
            cv::Mat image = cv::imread(imageFile);

            if (image.empty()) {
                std::cerr << "Failed to load image: " << imageFile << std::endl;
                continue;
            }

            // Process the frame through the TrackerSystem
            auto results = system.processFrame(image);

            // Draw the tracking results
            for (const auto& [mean_state, trackId, score, particles] : results) 
            {
                // Extract mean state components
                float mean_x = mean_state[0]; // Position x
                float mean_y = mean_state[1]; // Position y
                float mean_w = mean_state[4]; // Width
                float mean_h = mean_state[5]; // Height

                // Draw the position as a blue circle
                cv::circle(image, cv::Point(static_cast<int>(mean_x), static_cast<int>(mean_y)), 5, cv::Scalar(255, 0, 0), -1); // Blue circle

                // Draw the width and height as a rectangle
                cv::Rect box(static_cast<int>(mean_x - mean_w / 2), static_cast<int>(mean_y - mean_h / 2), 
                            static_cast<int>(mean_w), static_cast<int>(mean_h));
                cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2); // Green rectangle

                // Add track ID and score as a label
                std::ostringstream oss;
                oss.precision(2);
                oss << std::fixed << "ID: " << trackId << " | Score: " << score;
                cv::putText(image, oss.str(), cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

                // Create an overlay for transparent particles
                cv::Mat overlay;
                image.copyTo(overlay);

                // Draw particles as small circles on the overlay
                for (const auto& particle : particles) 
                {
                    if (particle[0] > 0 && particle[0] < image.cols - 1 && particle[1] > 0 && particle[1] < image.rows - 1) 
                    {
                        cv::circle(overlay, cv::Point(static_cast<int>(particle[0]), static_cast<int>(particle[1])), 
                                2, cv::Scalar(0, 0, 255, 128), -1); // Red with transparency
                    }
                }

                // Blend the overlay with the original image
                double alpha = 0.5; // Transparency factor
                cv::addWeighted(overlay, alpha, image, 1 - alpha, 0, image);
            }

            // Update the progress in the same line
            std::cout << "Processed Frame: " << frame_count << "/" << imageFiles.size() << std::endl; 
            frame_count++;

            // Write the frame to the video
            videoWriter.write(image);
        }

         videoWriter.release();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
