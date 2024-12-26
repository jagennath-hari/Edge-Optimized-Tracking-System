#include "perception/engine.hpp"

namespace perception 
{

    TritonEngine::TritonEngine()
    {
        // Initialize Triton client
        triton::client::Error status = triton::client::InferenceServerGrpcClient::Create(&(this->tritonClient_), "localhost:8001");

        if (!status.IsOk()) throw std::runtime_error("Triton client initialization failed: " + status.Message());

        std::cout << "Connected to Triton server successfully!" << std::endl;
    }

    float TritonEngine::fp16ToFloat_(uint16_t fp16)
    {
        uint32_t t1 = fp16 & 0x7fff;            // Non-sign bits
        uint32_t t2 = fp16 & 0x8000;            // Sign bit
        uint32_t t3 = fp16 & 0x7c00;            // Exponent
        t1 <<= 13;                              // Align mantissa on MSB
        t2 <<= 16;                              // Shift sign bit into position
        t1 += 0x38000000;                       // Adjust bias
        t1 = (t3 == 0 ? 0 : t1);                // Denormals-as-zero
        t1 |= t2;                               // Re-insert sign bit
        float f;
        memcpy(&f, &t1, sizeof(f));             // Return as float
        return f;
    }

    triton::client::InferInput* TritonEngine::createTritonInput(const std::string& name, const cv::Mat& image, const std::string& datatype)
    {
        std::vector<int64_t> shape = {image.rows, image.cols, image.channels()};
        triton::client::InferInput* input = nullptr;

        triton::client::Error status = triton::client::InferInput::Create(&input, name, shape, datatype);
        if (!status.IsOk()) throw std::runtime_error("Triton input creation failed: " + status.Message());

        status = input->AppendRaw(image.data, image.total() * image.elemSize());
        if (!status.IsOk()) 
        {
            delete input;
            throw std::runtime_error("Failed to append raw data to Triton input: " + status.Message());
        }

        return input;
    }

    std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> TritonEngine::extractOutputs_(const triton::client::InferResult* result) const
    {
        const int32_t* boxes_raw = nullptr;
        const uint16_t* scores_raw = nullptr;
        const int32_t* ids_raw = nullptr;
        size_t boxes_byte_size = 0, scores_byte_size = 0, ids_byte_size = 0;

        triton::client::Error status = result->RawData("boxes", reinterpret_cast<const uint8_t**>(&boxes_raw), &boxes_byte_size);
        if (!status.IsOk()) throw std::runtime_error("Bouding box Error: " + status.Message());

        status = result->RawData("scores", reinterpret_cast<const uint8_t**>(&scores_raw), &scores_byte_size);
        if (!status.IsOk()) throw std::runtime_error("Score Error: " + status.Message());

        status = result->RawData("class_ids", reinterpret_cast<const uint8_t**>(&ids_raw), &ids_byte_size);
        if (!status.IsOk()) throw std::runtime_error("Class ID Error: " + status.Message());

        size_t num_boxes = boxes_byte_size / (4 * sizeof(int32_t));
        size_t num_scores = scores_byte_size / sizeof(uint16_t);
        size_t num_ids = ids_byte_size / sizeof(int32_t);

        if (num_boxes != num_scores || num_boxes != num_ids) throw std::runtime_error("Output size mismatch: boxes=" + std::to_string(num_boxes) + ", scores=" + std::to_string(num_scores) + ", ids=" + std::to_string(num_ids));

        // Convert raw data into usable objects
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> classIds;

        boxes.reserve(num_boxes);
        scores.reserve(num_scores);
        classIds.reserve(num_ids);

        for (size_t i = 0; i < num_boxes; ++i) 
        {
            // Extract box coordinates
            int x1 = boxes_raw[i * 4 + 0];
            int y1 = boxes_raw[i * 4 + 1];
            int x2 = boxes_raw[i * 4 + 2];
            int y2 = boxes_raw[i * 4 + 3];

            // Convert FP16 scores to float
            float score = this->fp16ToFloat_(scores_raw[i]);

            boxes.emplace_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
            scores.push_back(score);
            classIds.push_back(ids_raw[i]);
        }

        return std::make_tuple(std::move(boxes), std::move(scores), std::move(classIds));
    }

    std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> TritonEngine::infer(const cv::Mat& inputImage)
    {
        // Timing start (only in debug mode)
        #ifdef DEBUG
            std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
        #endif

        std::unique_ptr<triton::client::InferInput> triton_input;
        triton_input.reset(this->createTritonInput("input_image", inputImage, "UINT8"));
        std::vector<triton::client::InferInput*> inputs = {triton_input.get()};
        std::unique_ptr<triton::client::InferRequestedOutput> boxes;
        std::unique_ptr<triton::client::InferRequestedOutput> scores;
        std::unique_ptr<triton::client::InferRequestedOutput> class_ids;
        // Create Triton outputs
        triton::client::InferRequestedOutput* temp = nullptr;

        triton::client::InferRequestedOutput::Create(&temp, "boxes");
        boxes.reset(temp);

        triton::client::InferRequestedOutput::Create(&temp, "scores");
        scores.reset(temp);

        triton::client::InferRequestedOutput::Create(&temp, "class_ids");
        class_ids.reset(temp);

        // Wrap in a vector for inference
        std::vector<const triton::client::InferRequestedOutput*> outputs = {boxes.get(), scores.get(), class_ids.get()};

        // Perform inference
        triton::client::InferResult* result = nullptr;
        triton::client::Error status = this->tritonClient_->Infer(&result, triton::client::InferOptions("perception_ensembled"), inputs, outputs);

        if (!status.IsOk()) throw std::runtime_error("Inference Error: " + status.Message());
        std::unique_ptr<triton::client::InferResult> result_ptr(result);

        // Timing end (only in debug mode)
        #ifdef DEBUG
            std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
            std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "[DEBUG] Inference time: " << duration.count() << " ms" << std::endl;
        #endif

        return this->extractOutputs_(result_ptr.get());
    }

} // namespace perception
