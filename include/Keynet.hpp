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

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <cmath>

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

namespace DropD {

class KeyNetInference {
public:
    /**
     * @brief Construct a new Key Net Inference object
     * 
     * @param detector_model_path Path to the ONNX detector model
     * @param num_levels Number of pyramid levels for multi-scale detection
     * @param scale_factor Scale factor between pyramid levels
     * @param nms_size Size of the NMS window
     * @param nms_threshold Threshold for NMS
     */
    KeyNetInference(const std::string& detector_model_path, 
                   int num_levels = 1,
                   float scale_factor = 1.2,
                   int nms_size = 15,
                   float nms_threshold = 1.124f);

    /**
     * @brief Extract keypoints from an image
     * 
     * @param input_image Input image (grayscale or RGB)
     * @param keypoints Output vector to store detected keypoints
     * @param max_keypoints Maximum number of keypoints to detect
     */
    void extractFeatures(const cv::Mat& input_image, 
                        std::vector<cv::KeyPoint>& keypoints,
                        int max_keypoints = 5000);
    void extractFeaturesAtLevel(
        const cv::Mat &image, std::vector<cv::KeyPoint> &level_keypoints,
        int num_points_level, const cv::Size &original_size, bool is_upsampled,
        int level_idx);
private:
    // Configuration parameters matching training specs
    static const int NUM_FILTERS = 8;        // M = 8 filters as per training
    static const int KERNEL_SIZE = 5;        // 5x5 kernel size as per training
    int num_levels;                          // Number of pyramid levels
    float scale_factor;                      // Scale factor between levels
    int nms_size;                           // NMS window size
    float nms_threshold;                    // NMS threshold

    // ONNX Runtime members
    Ort::Env env;
    std::unique_ptr<Ort::Session> detector_session;
    Ort::MemoryInfo memory_info;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    // Helper functions
    cv::Mat removeBorders(const cv::Mat& score_map, int borders);
    cv::Mat customPyrDown(const cv::Mat& input, float factor);
    cv::Mat customPyrUp(const cv::Mat &input, float factor);
    std::vector<cv::KeyPoint> performNMS(const cv::Mat& score_map, int nms_size, float threshold);
};

} // namespace DropD
