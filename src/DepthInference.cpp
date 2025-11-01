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

#include "DepthInference.h"
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cstring>
#include <cmath>
#include <onnxruntime_cxx_api.h>

namespace DropD {

// ---- DepthEstimationInference ----
DepthEstimationInference::DepthEstimationInference(const std::string& model_name)
    : env(ORT_LOGGING_LEVEL_WARNING, model_name.c_str()) {
}


cv::Mat DepthEstimationInference::filterDepthByDistance(const cv::Mat& depth_map, float max_distance_meters) {
    cv::Mat filtered_depth = depth_map.clone();
    cv::Mat mask = depth_map > max_distance_meters;
    filtered_depth.setTo(0.0f, mask);
    return filtered_depth;
}

cv::Mat DepthEstimationInference::filterDepthByRange(const cv::Mat& depth_map, float min_distance_meters, float max_distance_meters) {
    cv::Mat filtered_depth = depth_map.clone();
    cv::Mat mask_min = depth_map < min_distance_meters;
    cv::Mat mask_max = depth_map > max_distance_meters;
    cv::Mat combined_mask;
    cv::bitwise_or(mask_min, mask_max, combined_mask);
    filtered_depth.setTo(0.0f, combined_mask);
    return filtered_depth;
}

void DepthEstimationInference::initializeSession(const std::string& model_path) {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.EnableMemPattern();
    session_options.EnableCpuMemArena();
    
    try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.gpu_mem_limit = SIZE_MAX;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.do_copy_in_default_stream = 1;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    } catch (const std::exception&) {
        // Fallback to CPU
    }
    
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
}
void DepthEstimationInference::setupInputOutputInfo() {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session->GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = session->GetInputNameAllocated(i, allocator);
        input_names.push_back(input_name.get());
        input_names_allocated.push_back(std::move(input_name));
    }
    Ort::TypeInfo input_type_info = session->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape = input_tensor_info.GetShape();
    fixDynamicDimensions();
    size_t num_output_nodes = session->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = session->GetOutputNameAllocated(i, allocator);
        output_names.push_back(output_name.get());
        output_names_allocated.push_back(std::move(output_name));
    }
}

void DepthEstimationInference::fixDynamicDimensions() {
    if (input_shape[0] == -1) input_shape[0] = 1;
    if (input_shape[2] == -1) input_shape[2] = 518;
    if (input_shape[3] == -1) input_shape[3] = 518;
}

// printModelInfo removed for production build

std::vector<float> DepthEstimationInference::matToVector(const cv::Mat& mat) {
    std::vector<float> data;
    std::vector<cv::Mat> channels(3);
    cv::split(mat, channels);
    for (int c = 0; c < 3; c++) {
        cv::Mat channel = channels[c];
        data.insert(data.end(), (float*)channel.data, (float*)channel.data + channel.total());
    }
    return data;
}

cv::Mat DepthEstimationInference::runInference(const cv::Mat& image) {
    int original_height = image.rows;
    int original_width = image.cols;

    cv::Mat processed_image = preprocessImage(image);
    std::vector<float> input_data = matToVector(processed_image);
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), 
        input_shape.data(), input_shape.size());
    
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    
    Ort::RunOptions run_options;
    auto output_tensors = session->Run(run_options, 
                                     input_names.data(), input_tensors.data(), 1,
                                     output_names.data(), output_names.size());
    
    cv::Mat depth_map = extractDepthFromOutput(output_tensors);
    cv::Mat result = postprocessDepth(depth_map, original_width, original_height);
    
    // Apply filtering if max_depth_distance is set
    if (max_depth_distance_ > 0.0f) {
        result = filterDepthByDistance(result, max_depth_distance_);
    }
    
    return result;
}

cv::Mat DepthEstimationInference::colorizeDepth(const cv::Mat& depth_map, int colormap) {
    cv::Mat colored, normalized_depth;
    double min_val, max_val;
    cv::minMaxLoc(depth_map, &min_val, &max_val);
    if (max_val == 0.0) {
        normalized_depth = cv::Mat::zeros(depth_map.size(), CV_8U);
    } else {
        depth_map.convertTo(normalized_depth, CV_8U, 255.0 / max_val);
    }
    cv::applyColorMap(normalized_depth, colored, colormap);
    return colored;
}

void DepthEstimationInference::saveDepthMap(const cv::Mat& depth_map, const std::string& output_path) {
    cv::imwrite(output_path, depth_map);
}


// ---- DepthAnythingV2Inference NOT USED FOR NOW ----
DepthAnythingV2Inference::DepthAnythingV2Inference(const std::string& model_path)
    : DepthEstimationInference("DepthAnythingV2Inference") {
    initializeSession(model_path);
    setupInputOutputInfo();
    
    // Warmup inference
    cv::Mat dummy_image(input_shape[2], input_shape[3], CV_8UC3, cv::Scalar(128, 128, 128));
    try {
        runInference(dummy_image);
    } catch (...) {
        // Warmup failed, continue anyway
    }
}

cv::Mat DepthAnythingV2Inference::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;
    cv::cvtColor(image, processed, cv::COLOR_BGR2RGB);
    int target_height = input_shape[2];
    int target_width = input_shape[3];
    cv::resize(processed, processed, cv::Size(target_width, target_height), 0, 0, cv::INTER_CUBIC);
    processed.convertTo(processed, CV_32F, 1.0/255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(processed, channels);
    for (int c = 0; c < 3; c++) channels[c] = (channels[c] - mean[c]) / std[c];
    cv::merge(channels, processed);
    return processed;
}

cv::Mat DepthAnythingV2Inference::extractDepthFromOutput(std::vector<Ort::Value>& output_tensors) {
    auto actual_output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    float* depth_data = output_tensors[0].GetTensorMutableData<float>();
    cv::Mat depth_map;
    if (actual_output_shape.size() == 4) {
        int out_height = actual_output_shape[2];
        int out_width = actual_output_shape[3];
        depth_map = cv::Mat(out_height, out_width, CV_32F, depth_data);
    } else if (actual_output_shape.size() == 3) {
        int out_height = actual_output_shape[1];
        int out_width = actual_output_shape[2];
        depth_map = cv::Mat(out_height, out_width, CV_32F, depth_data);
    } else {
        throw std::runtime_error("Unexpected output shape dimensions");
    }
    return depth_map.clone();
}

cv::Mat DepthAnythingV2Inference::postprocessDepth(const cv::Mat& depth_map, int target_width, int target_height) {
    cv::Mat processed_depth;
    cv::resize(depth_map, processed_depth, cv::Size(target_width, target_height), 0, 0, cv::INTER_CUBIC);
    return processed_depth;
}

cv::Mat DepthAnythingV2Inference::infer(const cv::Mat& image) {
    return runInference(image);
}

// ---- UniDepthInference ----
UniDepthInference::UniDepthInference(const std::string& model_path, float fx, float fy, float cx, float cy, int width, int height)
    : DepthEstimationInference("UniDepthInference"), use_camera_input_(false), 
      original_width_(640), original_height_(480) {
    initializeSession(model_path);
    setupInputOutputInfo();
    setupUniDepthOutputShapes();
    
    // Check if model expects camera input
    if (input_names.size() == 2) {
        use_camera_input_ = true;
        // Use provided intrinsics if valid, otherwise use TUM1 defaults
        if (fx > 0 && fy > 0 && width > 0 && height > 0) {
            setCameraIntrinsics(fx, fy, cx, cy, width, height);
            std::cout << "✓ Using camera intrinsics from config:" << std::endl;
            std::cout << "  fx=" << fx << ", fy=" << fy << ", cx=" << cx << ", cy=" << cy << std::endl;
            std::cout << "  Resolution: " << width << "x" << height << std::endl;
        } else {
            // Fallback to TUM1 camera intrinsics (native resolution 640x480)
            setCameraIntrinsics(517.306408f, 516.469215f, 318.643040f, 255.313989f, 640, 480);
            std::cout << "✓ Using default TUM1 camera intrinsics:" << std::endl;
            std::cout << "  fx=517.3, fy=516.5, cx=318.6, cy=255.3" << std::endl;
            std::cout << "  Resolution: 640x480" << std::endl;
        }
    }
    
    // Warmup inference
    cv::Mat dummy_image(input_shape[2], input_shape[3], CV_8UC3, cv::Scalar(128, 128, 128));
    try {
        runInference(dummy_image);
    } catch (...) {
        // Warmup failed, continue anyway
    }
}

void UniDepthInference::setCameraIntrinsics(const cv::Mat& camera_matrix, int original_width, int original_height) {
    if (camera_matrix.rows != 3 || camera_matrix.cols != 3) {
        throw std::runtime_error("Camera matrix must be 3x3");
    }
    camera_matrix_ = camera_matrix.clone();
    original_width_ = original_width;
    original_height_ = original_height;
}

void UniDepthInference::setCameraIntrinsics(float fx, float fy, float cx, float cy, int original_width, int original_height) {
    camera_matrix_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    original_width_ = original_width;
    original_height_ = original_height;
}

std::vector<float> UniDepthInference::buildCameraRays(int height, int width, const cv::Mat& K) {
    // Extract original intrinsics (for native camera resolution)
    double fx_orig = K.at<double>(0, 0);
    double fy_orig = K.at<double>(1, 1);
    double cx_orig = K.at<double>(0, 2);
    double cy_orig = K.at<double>(1, 2);
    
    // Scale intrinsics to match input resolution
    double scale_w = static_cast<double>(width) / original_width_;
    double scale_h = static_cast<double>(height) / original_height_;
    double fx = fx_orig * scale_w;
    double fy = fy_orig * scale_h;
    double cx = cx_orig * scale_w;
    double cy = cy_orig * scale_h;
    
    // Allocate buffer for rays: [1, 3, height, width]
    std::vector<float> rays(3 * height * width);
    
    // Generate normalized ray directions for each pixel
    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            int idx = v * width + u;
            
            // Compute ray direction in camera space (using scaled intrinsics)
            float dir_x = (u - cx) / fx;
            float dir_y = (v - cy) / fy;
            float dir_z = 1.0f;
            
            // Normalize ray direction
            float norm = std::sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
            
            // Store in CHW format: [3, H, W]
            rays[0 * height * width + idx] = dir_x / norm;  // X channel
            rays[1 * height * width + idx] = dir_y / norm;  // Y channel
            rays[2 * height * width + idx] = dir_z / norm;  // Z channel
        }
    }
    
    return rays;
}

void UniDepthInference::fixDynamicDimensions() {
    if (input_shape[0] == -1) input_shape[0] = 1;
    if (input_shape[2] == -1) input_shape[2] = 354;  // Reduced resolution for speed without loss of accuracy
    if (input_shape[3] == -1) input_shape[3] = 490;  // Reduced resolution for speed
}

void UniDepthInference::setupUniDepthOutputShapes() {
    Ort::TypeInfo output_type_info_0 = session->GetOutputTypeInfo(0);
    auto output_tensor_info_0 = output_type_info_0.GetTensorTypeAndShapeInfo();
    output_shape_pts3d = output_tensor_info_0.GetShape();
    Ort::TypeInfo output_type_info_1 = session->GetOutputTypeInfo(1);
    auto output_tensor_info_1 = output_type_info_1.GetTensorTypeAndShapeInfo();
    output_shape_confidence = output_tensor_info_1.GetShape();
    Ort::TypeInfo output_type_info_2 = session->GetOutputTypeInfo(2);
    auto output_tensor_info_2 = output_type_info_2.GetTensorTypeAndShapeInfo();
    output_shape_intrinsics = output_tensor_info_2.GetShape();
}

cv::Mat UniDepthInference::preprocessImage(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty in preprocessImage!");
    }
    if (input_shape.size() < 4) {
        throw std::runtime_error("input_shape has less than 4 elements!");
    }
    int target_height = input_shape[2];  // Will be 354 (down from 480)
    int target_width = input_shape[3];   // Will be 490 (down from 640)
    if (target_height <= 0 || target_width <= 0) {
        throw std::runtime_error("Invalid target size in preprocessImage!");
    }
    cv::Mat processed;
    cv::cvtColor(image, processed, cv::COLOR_BGR2RGB);
    
    // Use INTER_AREA for downscaling (better quality than INTER_LINEAR)
    int interpolation = (image.rows > target_height || image.cols > target_width) 
                       ? cv::INTER_AREA  // Downscaling
                       : cv::INTER_LINEAR;  // Upscaling (rare)
    cv::resize(processed, processed, cv::Size(target_width, target_height), 0, 0, interpolation);
    
    processed.convertTo(processed, CV_32F, 1.0/255.0);
    return processed;
}

cv::Mat UniDepthInference::extractDepthFromOutput(std::vector<Ort::Value>& output_tensors) {
    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
        throw std::runtime_error("No tensor output from ONNX inference!");
    }
    auto pts3d_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    if (pts3d_shape.size() != 4 || pts3d_shape[1] != 3) {
        throw std::runtime_error("Unexpected pts_3d output shape!");
    }
    float* pts3d_data = output_tensors[0].GetTensorMutableData<float>();
    int height = pts3d_shape[2];
    int width = pts3d_shape[3];
    cv::Mat depth_map(height, width, CV_32F);
    for (int i = 0; i < height * width; i++) {
        depth_map.at<float>(i / width, i % width) = pts3d_data[2 * height * width + i];
    }
    return depth_map;
}

cv::Mat UniDepthInference::postprocessDepth(const cv::Mat& depth_map, int target_width, int target_height) {
    cv::Mat processed_depth;
    cv::resize(depth_map, processed_depth, cv::Size(target_width, target_height));
    return processed_depth;
}

cv::Mat UniDepthInference::runInference(const cv::Mat& image) {
    int original_height = image.rows;
    int original_width = image.cols;

    cv::Mat processed_image = preprocessImage(image);
    std::vector<float> input_data = matToVector(processed_image);
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    std::vector<Ort::Value> input_tensors;
    
    // Add RGB tensor
    Ort::Value rgb_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), 
        input_shape.data(), input_shape.size());
    input_tensors.push_back(std::move(rgb_tensor));
    
    // If camera-aware model, add rays tensor
    std::vector<float> rays_data;
    if (use_camera_input_) {
        if (camera_matrix_.empty()) {
            throw std::runtime_error("Camera intrinsics not set!");
        }
        
        int target_height = input_shape[2];
        int target_width = input_shape[3];
        
        rays_data = buildCameraRays(target_height, target_width, camera_matrix_);
        
        std::vector<int64_t> rays_shape = {1, 3, target_height, target_width};
        Ort::Value rays_tensor = Ort::Value::CreateTensor<float>(
            memory_info, rays_data.data(), rays_data.size(),
            rays_shape.data(), rays_shape.size());
        input_tensors.push_back(std::move(rays_tensor));
    }
    
    Ort::RunOptions run_options;
    auto output_tensors = session->Run(run_options, 
                                     input_names.data(), input_tensors.data(), input_tensors.size(),
                                     output_names.data(), output_names.size());
    
    cv::Mat depth_map = extractDepthFromOutput(output_tensors);
    cv::Mat result = postprocessDepth(depth_map, original_width, original_height);
    
    // Apply filtering if max_depth_distance is set
    if (max_depth_distance_ > 0.0f) {
        result = filterDepthByDistance(result, max_depth_distance_);
    }
    
    return result;
}


cv::Mat UniDepthInference::infer(const cv::Mat& image) {
    return runInference(image);
}

// ---- Factory ----
std::unique_ptr<DepthEstimationInference> createDepthEstimator(const std::string& model_type, 
                                                              const std::string& model_path,
                                                              float fx, float fy, 
                                                              float cx, float cy,
                                                              int width, int height) {
    if (model_type == "depthanything" || model_type == "depthanythingv2") {
        return std::make_unique<DepthAnythingV2Inference>(model_path);
    } else if (model_type == "unidepth") {
        return std::make_unique<UniDepthInference>(model_path, fx, fy, cx, cy, width, height);
    } else {
        throw std::invalid_argument("Unknown model type: " + model_type);
    }
}

} // namespace DropD