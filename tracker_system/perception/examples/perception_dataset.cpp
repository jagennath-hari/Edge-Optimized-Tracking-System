#include "perception/engine.hpp"
#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <memory>
#include <sstream>

int main(int argc, char* argv[]) 
{
    // Check if the user provided a dataset directory
    if (argc != 2) 
    {
        std::cerr << "Usage: " << argv[0] << " <dataset_directory>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string datasetDirectory = argv[1];

    try 
    {
        namespace fs = std::filesystem;

        // Verify the directory exists
        if (!fs::exists(datasetDirectory) || !fs::is_directory(datasetDirectory)) 
        {
            throw std::runtime_error("Invalid directory: " + datasetDirectory);
        }

        // Collect and sort image files
        std::vector<std::string> imageFiles;
        for (const fs::directory_entry& entry : fs::directory_iterator(datasetDirectory)) 
        {
            if (entry.is_regular_file()) 
            {
                const std::string extension = entry.path().extension().string();
                if (extension == ".jpg" || extension == ".png" || extension == ".jpeg") 
                {
                    imageFiles.push_back(entry.path().string());
                }
            }
        }

        std::sort(imageFiles.begin(), imageFiles.end());

        if (imageFiles.empty()) 
        {
            throw std::runtime_error("No valid image files found in directory: " + datasetDirectory);
        }

        // Instantiate the TritonEngine as a unique pointer
        std::unique_ptr<perception::TritonEngine> engine = std::make_unique<perception::TritonEngine>();

        // Loop through the images and perform inference
        for (const std::string& imageFile : imageFiles) 
        {
            cv::Mat image = cv::imread(imageFile);

            if (image.empty()) 
            {
                std::cerr << "Failed to load image: " << imageFile << std::endl;
                continue;
            }

            // Perform inference using the TritonEngine
            auto [boxes, scores, classIds] = engine->infer(image);

            // Draw the results on the image
            for (size_t i = 0; i < boxes.size(); ++i) 
            {
                const cv::Rect& box = boxes[i];
                const float score = scores[i];
                const int classId = classIds[i];

                cv::Scalar color = (classId == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                cv::rectangle(image, box, color, 2);

                // Format the score to two decimal places
                std::ostringstream oss;
                oss.precision(2);
                oss << std::fixed << score;
                std::string formattedScore = oss.str();

                // Construct the label
                std::string label = "ID: " + std::to_string(classId) + ", Score: " + formattedScore;
                cv::putText(image, label, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            }

            // Display the image
            cv::imshow("Image Viewer", image);

            // Wait for a key press or quit on 'q'
            if (cv::waitKey(20) == 'q') break;
        }

        cv::destroyAllWindows();
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
