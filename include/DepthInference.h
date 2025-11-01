/**
 * This file is part of DropD-SLAM
 *
 * Copyright (C) 2025 Alican Karaomer
 * Email: alicaank14@gmail.com
 *
 * DropD-SLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DropD-SLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DropD-SLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <string>

namespace DropD {

// Abstract base class for depth estimation inference
class DepthEstimationInference {
protected:
    std::unique_ptr<Ort::Session> session;
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<Ort::AllocatedStringPtr> input_names_allocated;
    std::vector<Ort::AllocatedStringPtr> output_names_allocated;
    std::vector<int64_t> input_shape;
    float max_depth_distance_ = 0.0f;

public:
    DepthEstimationInference(const std::string& model_name);
    virtual ~DepthEstimationInference() = default;
    
    // Depth filtering control
    void setMaxDepthDistance(float max_distance) { max_depth_distance_ = max_distance; }
    float getMaxDepthDistance() const { return max_depth_distance_; }

    virtual cv::Mat preprocessImage(const cv::Mat& image) = 0;
    virtual cv::Mat postprocessDepth(const cv::Mat& depth_map, int target_width, int target_height) = 0;
    virtual cv::Mat extractDepthFromOutput(std::vector<Ort::Value>& output_tensors) = 0;

    cv::Mat filterDepthByDistance(const cv::Mat& depth_map, float max_distance_meters);
    cv::Mat filterDepthByRange(const cv::Mat& depth_map, float min_distance_meters, float max_distance_meters);

    void initializeSession(const std::string& model_path);
    virtual void fixDynamicDimensions();
    void setupInputOutputInfo();
    std::vector<float> matToVector(const cv::Mat& mat);

    virtual cv::Mat runInference(const cv::Mat& image);
    cv::Mat colorizeDepth(const cv::Mat& depth_map, int colormap = cv::COLORMAP_PLASMA);
    void saveDepthMap(const cv::Mat& depth_map, const std::string& output_path);

    // Performance helpers removed for production build
};

// Depth-Anything-V2 implementation
class DepthAnythingV2Inference : public DepthEstimationInference {
private:
    const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    const std::vector<float> std = {0.229f, 0.224f, 0.225f};
public:
    DepthAnythingV2Inference(const std::string& model_path);
    cv::Mat preprocessImage(const cv::Mat& image) override;
    cv::Mat extractDepthFromOutput(std::vector<Ort::Value>& output_tensors) override;
    cv::Mat postprocessDepth(const cv::Mat& depth_map, int target_width, int target_height) override;
    cv::Mat infer(const cv::Mat& image);
};

// UniDepth implementation
class UniDepthInference : public DepthEstimationInference {
private:
    std::vector<int64_t> output_shape_pts3d;
    std::vector<int64_t> output_shape_confidence;
    std::vector<int64_t> output_shape_intrinsics;
    
    // Camera intrinsics (optional - for camera-aware models)
    bool use_camera_input_;
    cv::Mat camera_matrix_;  // 3x3 intrinsics matrix
    int original_width_;     // Original camera resolution width
    int original_height_;    // Original camera resolution height
    
    // Build camera rays from intrinsics (scaled to input resolution)
    std::vector<float> buildCameraRays(int height, int width, const cv::Mat& K);
    
public:
    struct InferenceResult {
        cv::Mat depth;
        cv::Mat confidence;
        cv::Mat intrinsics;
    };
    
    UniDepthInference(const std::string& model_path, float fx = 0, float fy = 0, float cx = 0, float cy = 0, int width = 0, int height = 0);
    
    // Set camera intrinsics (for --with-camera models)
    // original_width and original_height are the camera's native resolution
    void setCameraIntrinsics(const cv::Mat& camera_matrix, int original_width = 640, int original_height = 480);
    void setCameraIntrinsics(float fx, float fy, float cx, float cy, int original_width = 640, int original_height = 480);
    
    void fixDynamicDimensions() override;
    void setupUniDepthOutputShapes();
    cv::Mat preprocessImage(const cv::Mat& image) override;
    cv::Mat extractDepthFromOutput(std::vector<Ort::Value>& output_tensors) override;
    cv::Mat postprocessDepth(const cv::Mat& depth_map, int target_width, int target_height) override;
    
    // Inference with camera support (overrides base class)
    cv::Mat runInference(const cv::Mat& image) override;
    cv::Mat infer(const cv::Mat& image);
};

// Factory function
std::unique_ptr<DepthEstimationInference> createDepthEstimator(const std::string& model_type, 
                                                              const std::string& model_path,
                                                              float fx = 0, float fy = 0, 
                                                              float cx = 0, float cy = 0,
                                                              int width = 0, int height = 0);

} // namespace DropD