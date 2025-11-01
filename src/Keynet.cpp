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

#include "Keynet.hpp"

namespace DropD {

KeyNetInference::KeyNetInference(const std::string &detector_model_path,
                                 int num_levels, float scale_factor,
                                 int nms_size, float nms_threshold)
    : num_levels(num_levels), scale_factor(scale_factor), nms_size(nms_size),
      nms_threshold(nms_threshold),
      env(ORT_LOGGING_LEVEL_WARNING, "KeyNetInference"),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPUOutput)),
      input_names{"input"}, output_names{"output"} {

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetInterOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  try {
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    cuda_options.arena_extend_strategy = 0;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_options.do_copy_in_default_stream = 1;
    session_options.AppendExecutionProvider_CUDA(cuda_options);
  } catch (const std::exception&) {
    // Fallback to CPU
  }

  detector_session = std::make_unique<Ort::Session>(env, detector_model_path.c_str(), session_options);

  // Warmup inference
  const int warmup_size = 64;
  cv::Mat dummy_image(warmup_size, warmup_size, CV_8UC1, cv::Scalar(128));
  cv::Mat float_image;
  dummy_image.convertTo(float_image, CV_32F, 1.0 / 255.0);

  std::vector<float> input_tensor_values(float_image.total());
  memcpy(input_tensor_values.data(), float_image.ptr<float>(),
         float_image.total() * sizeof(float));

  std::vector<int64_t> input_dims = {1, 1,
                                     static_cast<int64_t>(float_image.rows),
                                     static_cast<int64_t>(float_image.cols)};

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_tensor_values.data(), input_tensor_values.size(),
      input_dims.data(), input_dims.size());

  detector_session->Run(Ort::RunOptions{nullptr}, input_names.data(),
                        &input_tensor, 1, output_names.data(), 1);
}

cv::Mat KeyNetInference::removeBorders(const cv::Mat &score_map, int borders) {
  cv::Mat result = score_map.clone();
  result.rowRange(0, borders) = 0;
  result.rowRange(result.rows - borders, result.rows) = 0;
  result.colRange(0, borders) = 0;
  result.colRange(result.cols - borders, result.cols) = 0;
  return result;
}

cv::Mat KeyNetInference::customPyrDown(const cv::Mat &input, float factor) {
  cv::Size new_size(cvRound(input.cols / factor), cvRound(input.rows / factor));
  cv::Mat resized;
  cv::Mat blurred;
  cv::GaussianBlur(input, blurred, cv::Size(5, 5), 0);
  cv::resize(blurred, resized, new_size, 0, 0, cv::INTER_LINEAR);
  return resized;
}

std::vector<cv::KeyPoint> KeyNetInference::performNMS(const cv::Mat &score_map,
                                                      int nms_size,
                                                      float threshold) {
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat dilated;
  cv::dilate(score_map, dilated, cv::Mat(), cv::Point(-1, -1), 1);

  // Compute local maxima
  cv::Mat maxima =
      (score_map == dilated); // Boolean mask where score == local max

  // Apply thresholding: Keep only maxima that are greater than threshold
  cv::Mat mask = (score_map > threshold) & maxima;

  for (int i = nms_size; i < score_map.rows - nms_size; i++) {
    for (int j = nms_size; j < score_map.cols - nms_size; j++) {
      float score = score_map.at<float>(i, j);
      if (mask.at<uchar>(i, j)) {
        cv::KeyPoint kp;
        kp.pt.x = j;
        kp.pt.y = i;
        kp.response = score;
        kp.octave = 0;
        keypoints.emplace_back(kp);
      }
    }
  }
  return keypoints;
}
void KeyNetInference::extractFeatures(const cv::Mat &input_image,
                                      std::vector<cv::KeyPoint> &keypoints,
                                      int max_keypoints) {
  if (input_image.empty()) {
    keypoints.clear();
    return;
  }

  cv::Mat gray_image;
  if (input_image.channels() == 3) {
    cv::cvtColor(input_image, gray_image, cv::COLOR_BGR2GRAY);
  } else {
    gray_image = input_image.clone();
  }

  std::vector<cv::Mat> pyramid;
  pyramid.reserve(num_levels);
  pyramid.push_back(gray_image);

  for (int i = 0; i < num_levels; i++) {
    pyramid.push_back(customPyrDown(pyramid.back(), scale_factor));
  }

  keypoints.clear();
  keypoints.reserve(max_keypoints * num_levels);

  for (int level = 0; level < num_levels; level++) {
    cv::Mat current_image = pyramid[level];
    float current_scale = std::pow(scale_factor, level);

    cv::Mat float_image;
    current_image.convertTo(float_image, CV_32F, 1.0 / 255.0);

    std::vector<float> input_tensor_values(float_image.total());
    memcpy(input_tensor_values.data(), float_image.ptr<float>(),
           float_image.total() * sizeof(float));

    std::vector<int64_t> input_dims = {1, 1,
                                       static_cast<int64_t>(float_image.rows),
                                       static_cast<int64_t>(float_image.cols)};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_dims.data(), input_dims.size());

    auto output_tensors =
        detector_session->Run(Ort::RunOptions{nullptr}, input_names.data(),
                              &input_tensor, 1, output_names.data(), 1);

    float *output_data = output_tensors[0].GetTensorMutableData<float>();
    cv::Mat score_map(float_image.size(), CV_32F, output_data);
    score_map = removeBorders(score_map, nms_size);
    std::vector<cv::KeyPoint> level_keypoints =
        performNMS(score_map, nms_size, nms_threshold);

    for (auto &kp : level_keypoints) {
      kp.pt.x *= current_scale;
      kp.pt.y *= current_scale;
      kp.size = 32.0f * current_scale;  // Default patch size
      kp.octave = level;
    }

    keypoints.insert(keypoints.end(), level_keypoints.begin(),
                     level_keypoints.end());
  }

  std::sort(keypoints.begin(), keypoints.end(),
            [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
              return a.response > b.response;
            });
  if (keypoints.size() > max_keypoints) {
    keypoints.resize(max_keypoints);
  }
}

} // namespace DropD

