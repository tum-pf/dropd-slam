/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

#include<opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "DepthInference.h" // <-- Add your unified inference header

#include<System.h>

using namespace std;
using namespace DropD;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

struct DepthJob {
    int frame_idx;
    cv::Mat rgb;
};

struct DepthResult {
    int frame_idx;
    cv::Mat depth;
};

std::queue<DepthJob> depth_jobs;
std::queue<DepthResult> depth_results;
std::mutex depth_mutex;
std::condition_variable depth_cv;
bool depth_thread_running = true;

void DepthWorker(std::unique_ptr<DepthEstimationInference>& depth_infer) {
    while (depth_thread_running) {
        DepthJob job;
        {
            std::unique_lock<std::mutex> lock(depth_mutex);
            depth_cv.wait(lock, []{ return !depth_jobs.empty() || !depth_thread_running; });
            if (!depth_thread_running) break;
            job = depth_jobs.front();
            depth_jobs.pop();
        }
        
        // Run inference
        cv::Mat depth = depth_infer->runInference(job.rgb);
        
        {
            std::lock_guard<std::mutex> lock(depth_mutex);
            depth_results.push({job.frame_idx, depth});
        }
    }
}

int main(int argc, char **argv)
{
    if(argc < 5 || argc > 7)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association [model_type model_path]" << endl;
        cerr << "Optional: model_type ('depthanything', 'unidepth') and model_path for monocular depth inference." << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Optional: Initialize monocular depth inference
    std::unique_ptr<DepthEstimationInference> depth_infer;
    bool use_mono_depth = false;
    string model_type = "";
    string model_path = "";
    float fx = 0, fy = 0, cx = 0, cy = 0;
    int cam_width = 0, cam_height = 0;
    float max_depth_dist = 0.0f;
    
    // Read from config file if available
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if(fsSettings.isOpened()) {
        cv::FileNode node = fsSettings["System.DepthModelType"];
        if(!node.empty() && node.isString()) {
            model_type = node.string();
        }
        node = fsSettings["System.DepthModelPath"];
        if(!node.empty() && node.isString()) {
            model_path = node.string();
        }
        
        // Read camera intrinsics
        node = fsSettings["Camera1.fx"];
        if(!node.empty()) fx = node.real();
        node = fsSettings["Camera1.fy"];
        if(!node.empty()) fy = node.real();
        node = fsSettings["Camera1.cx"];
        if(!node.empty()) cx = node.real();
        node = fsSettings["Camera1.cy"];
        if(!node.empty()) cy = node.real();
        node = fsSettings["Camera.width"];
        if(!node.empty()) cam_width = node.operator int();
        node = fsSettings["Camera.height"];
        if(!node.empty()) cam_height = node.operator int();
        
        // Read max depth distance for filtering
        node = fsSettings["System.MaxDepthDistance"];
        if(!node.empty()) max_depth_dist = node.real();
    }
    fsSettings.release();
    
    // Initialize depth inference if both type and path are available
    if(!model_type.empty() && !model_path.empty()) {
        depth_infer = createDepthEstimator(model_type, model_path, fx, fy, cx, cy, cam_width, cam_height);
        
        // Set max depth distance for filtering
        if(max_depth_dist > 0.0f) {
            depth_infer->setMaxDepthDistance(max_depth_dist);
            cout << "✓ Depth filtering enabled: max distance = " << max_depth_dist << " meters" << endl;
        } else {
            cout << "✓ Depth filtering disabled (no max distance set)" << endl;
        }
        
        use_mono_depth = true;
        
        // Additional warmup
        cv::Mat warmup_img = cv::imread(string(argv[3]) + "/" + vstrImageFilenamesRGB[0], cv::IMREAD_UNCHANGED);
        if(!warmup_img.empty()) {
            for (int i = 0; i < 3; ++i) {
                depth_infer->runInference(warmup_img);
            }
        }
    }

    std::thread depth_thread;
    if (use_mono_depth) {
        depth_thread_running = true;
        depth_thread = std::thread(DepthWorker, std::ref(depth_infer));
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD,true);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vector<float> vTimesDepth;  // Track depth inference times
    vTimesTrack.resize(nImages);
    vTimesDepth.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        // If depth is missing and monocular inference is enabled, generate depth
        if(use_mono_depth) {
            auto depth_start = std::chrono::high_resolution_clock::now();
            
            // Submit job
            {
                std::lock_guard<std::mutex> lock(depth_mutex);
                depth_jobs.push({ni, imRGB.clone()});
            }
            depth_cv.notify_one();

            // Wait for result for this frame
            cv::Mat depth;
            while (true) {
                std::unique_lock<std::mutex> lock(depth_mutex);
                if (!depth_results.empty() && depth_results.front().frame_idx == ni) {
                    imD = depth_results.front().depth;
                    depth_results.pop();
                    break;
                }
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            
            auto depth_end = std::chrono::high_resolution_clock::now();
            double depth_time = std::chrono::duration_cast<std::chrono::duration<double>>(depth_end - depth_start).count();
            vTimesDepth[ni] = depth_time;
        }

        if(imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

#ifdef COMPILEDWITHC14
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB,imD,tframe);

#ifdef COMPILEDWITHC14
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;
    
    // Depth inference statistics (if monocular depth was used)
    if(use_mono_depth) {
        sort(vTimesDepth.begin(),vTimesDepth.end());
        float totaldepthtime = 0;
        int depth_count = 0;
        for(int ni=0; ni<nImages; ni++)
        {
            if(vTimesDepth[ni] > 0) {
                totaldepthtime+=vTimesDepth[ni];
                depth_count++;
            }
        }
        if(depth_count > 0) {
            cout << "-------" << endl;
            cout << "Depth Inference Statistics:" << endl;
            cout << "median depth inference time: " << vTimesDepth[depth_count/2] << " s" << endl;
            cout << "mean depth inference time: " << totaldepthtime/depth_count << " s" << endl;
            cout << "mean depth inference time: " << (totaldepthtime/depth_count)*1000 << " ms" << endl;
            cout << "depth inference FPS: " << depth_count/totaldepthtime << endl;
            cout << "-------" << endl;
        }
    }
    
    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::duration<double> >(end_time - start_time).count();
    cout << "Total time: " << total_time << endl;
    cout << "FPS: " << nImages/total_time << endl;
    cout << "-------" << endl << endl;
    // Save camera trajectory
    SLAM.SaveTrajectoryTUM(string(argv[3])+"/OurCameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM(string(argv[3])+"/OurKeyFrameTrajectory.txt");

    // Join the depth thread if monocular depth inference was used
    if (use_mono_depth) {
        {
            std::lock_guard<std::mutex> lock(depth_mutex);
            depth_thread_running = false;
        }
        depth_cv.notify_one();
        depth_thread.join();
        
        // Explicitly destroy depth inference object before exiting
        // to ensure proper cleanup order and avoid double-free
        depth_infer.reset();
    }

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}