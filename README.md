# Dropping the D: RGB-D SLAM Without the Depth Sensor

**A real-time monocular SLAM system that achieves RGB-D accuracy without requiring a depth sensor.**
https://arxiv.org/abs/2510.06216
## Overview

**DropD-SLAM** replaces the active depth input of classical feature-based SLAM with pretrained deep learning modules:
- **Monocular Metric Depth Estimation** (UniDepthV2) - Predicts absolute scale depth from RGB
- **Learned Keypoint Detection** (KeyNet) - Robust feature detection and description
- **Instance Level Segmentation** (YOLO11) - Dynamic object detection and masking

The system removes keypoints on dynamic objects after mask dilation, assigns metric depth values to static keypoints from the predicted depth map, and backprojects them into 3D to initialize map points at absolute scale. These are provided to an unmodified **ORB-SLAM3 backend** for tracking, mapping, and loop closure.

### Key Features
- ✅ RGB-D accuracy from monocular RGB (no depth sensor required)
- ✅ Real-time performance on a single GPU
- ✅ Robust to dynamic objects via instance segmentation and mask dilation
- ✅ Metric absolute scale via monocular depth prediction
- ✅ Uses pretrained models (no retraining of the backend)

### Performance
On the **TUM RGB-D benchmark**, DropD-SLAM:
- Matches the accuracy of established RGB-D baselines on static sequences
- Outperforms recent monocular methods on dynamic sequences
- Operates in real-time on a single GPU

---


## Citation

If you use **DropD-SLAM** in an academic work, please cite:

```bibtex
@misc{kiray2025droppingdrgbdslam,
      title={Dropping the D: RGB-D SLAM Without the Depth Sensor}, 
      author={Mert Kiray and Alican Karaomer and Benjamin Busam},
      year={2025},
      eprint={2510.06216},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.06216}, 
}
```

## Original ORB-SLAM3

This work is built upon [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) (V1.0, December 22th, 2021).

**Original Authors:** Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, [José M. M. Montiel](http://webdiis.unizar.es/~josemari/), [Juan D. Tardos](http://webdiis.unizar.es/~jdtardos/).

ORB-SLAM3 is the first real-time SLAM library able to perform **Visual, Visual-Inertial and Multi-Map SLAM** with **monocular, stereo and RGB-D** cameras, using **pin-hole and fisheye** lens models.

This software is based on [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) developed by [Raul Mur-Artal](http://webdiis.unizar.es/~raulmur/), [Juan D. Tardos](http://webdiis.unizar.es/~jdtardos/), [J. M. M. Montiel](http://webdiis.unizar.es/~josemari/) and [Dorian Galvez-Lopez](http://doriangalvez.com/) ([DBoW2](https://github.com/dorian3d/DBoW2)).

### Related Publications:

[ORB-SLAM3] Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel and Juan D. Tardós, **ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM**, *IEEE Transactions on Robotics 37(6):1874-1890, Dec. 2021*. **[PDF](https://arxiv.org/abs/2007.11898)**.

[IMU-Initialization] Carlos Campos, J. M. M. Montiel and Juan D. Tardós, **Inertial-Only Optimization for Visual-Inertial Initialization**, *ICRA 2020*. **[PDF](https://arxiv.org/pdf/2003.05766.pdf)**

[ORBSLAM-Atlas] Richard Elvira, J. M. M. Montiel and Juan D. Tardós, **ORBSLAM-Atlas: a robust and accurate multi-map system**, *IROS 2019*. **[PDF](https://arxiv.org/pdf/1908.11585.pdf)**.

[ORBSLAM-VI] Raúl Mur-Artal, and Juan D. Tardós, **Visual-inertial monocular SLAM with map reuse**, IEEE Robotics and Automation Letters, vol. 2 no. 2, pp. 796-803, 2017. **[PDF](https://arxiv.org/pdf/1610.05949.pdf)**. 

[Stereo and RGB-D] Raúl Mur-Artal and Juan D. Tardós. **ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras**. *IEEE Transactions on Robotics,* vol. 33, no. 5, pp. 1255-1262, 2017. **[PDF](https://arxiv.org/pdf/1610.06475.pdf)**.

[Monocular] Raúl Mur-Artal, José M. M. Montiel and Juan D. Tardós. **ORB-SLAM: A Versatile and Accurate Monocular SLAM System**. *IEEE Transactions on Robotics,* vol. 31, no. 5, pp. 1147-1163, 2015. (**2015 IEEE Transactions on Robotics Best Paper Award**). **[PDF](https://arxiv.org/pdf/1502.00956.pdf)**.

[DBoW2 Place Recognition] Dorian Gálvez-López and Juan D. Tardós. **Bags of Binary Words for Fast Place Recognition in Image Sequences**. *IEEE Transactions on Robotics,* vol. 28, no. 5, pp. 1188-1197, 2012. **[PDF](http://doriangalvez.com/php/dl.php?dlp=GalvezTRO12.pdf)**

# 1. License and Citation

## License

DropD-SLAM is built upon ORB-SLAM3, which is released under [GPLv3 license](https://github.com/UZ-SLAMLab/ORB_SLAM3/LICENSE). For a list of all code/library dependencies (and associated licenses), please see [Dependencies.md](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/Dependencies.md).

For a closed-source version of ORB-SLAM3 for commercial purposes, please contact the authors.

## Citation


If you use ORB-SLAM3 in an academic work, please also cite:
  
    @article{ORBSLAM3_TRO,
      title={{ORB-SLAM3}: An Accurate Open-Source Library for Visual, Visual-Inertial 
               and Multi-Map {SLAM}},
      author={Campos, Carlos AND Elvira, Richard AND G\´omez, Juan J. AND Montiel, 
              Jos\'e M. M. AND Tard\'os, Juan D.},
      journal={IEEE Transactions on Robotics}, 
      volume={37},
      number={6},
      pages={1874-1890},
      year={2021}
     }

# 2. Prerequisites

We have tested the library in **Ubuntu 20.04** and **22.04**. A **NVIDIA GPU** with CUDA support is **required** for real-time depth estimation and dynamic object detection.


## Core Dependencies (ORB-SLAM3)

### C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

### Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Download and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

### OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Download and install instructions can be found at: http://opencv.org. **Required at least 4.5.0. Tested with OpenCV 4.5.4**.

```bash
sudo apt update
sudo apt install libopencv-dev python3-opencv
```

### Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

```bash
sudo apt install libeigen3-dev
```

### DBoW2 and g2o (Included in Thirdparty folder)
We use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

### Python
Required to calculate the alignment of the trajectory with the ground truth. **Required Numpy module**.

```bash
sudo apt install python3-dev python3-numpy
```

## DropD-SLAM Additional Dependencies

### CUDA and cuDNN
Required for GPU acceleration of deep learning models.

```bash
# Install CUDA 11.8 or 12.x (adjust version as needed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda

# Install cuDNN
# Follow instructions at: https://developer.nvidia.com/cudnn
```

### ONNX Runtime with CUDA
Required for running depth estimation, keypoint detection, and segmentation models.

```bash
# Download ONNX Runtime with GPU support (v1.16.0 or later)
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-gpu-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.16.3.tgz
sudo cp -r onnxruntime-linux-x64-gpu-1.16.3/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-gpu-1.16.3/lib/* /usr/local/lib/
sudo ldconfig
```

### PyTorch (for model export, optional)
Only needed if you want to export custom ONNX models.

```bash
# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### UniDepth (for depth estimation model, optional)
Clone and setup the UniDepth repository for model export.

```bash
cd /path/to/DropD-SLAM
git clone https://github.com/lpiccinelli-eth/UniDepth.git UniDepth
cd UniDepth
pip3 install -e .
```

# 3. Building DropD-SLAM

## Clone the Repository

```bash
git clone <repository-url> DropD-SLAM
cd DropD-SLAM
```

## Download Pretrained Models

DropD-SLAM requires three types of pretrained ONNX models:

### 1. Depth Estimation Model (UniDepthV2)

Download the UniDepthV2 ONNX model from the following link:

**Download Link:** [UniDepthV2 Model](https://drive.google.com/file/d/1MmL4UyKbwlOpIITj6Uh0SNv7TMtI0Ezy/view?usp=drive_link)

After downloading, place the model file in the `models/` directory:

```bash
# Move the downloaded model to models directory
# Replace 'unidepthv2_with_cam.onnx' with the actual filename from the download
mv /path/to/downloaded/unidepthv2_with_cam.onnx models/
```

**Note:** This model includes camera intrinsics support (`--with-camera` variant) and requires camera calibration parameters in your configuration file.


## Build the Library

We provide a script `build.sh` to build the *Thirdparty* libraries and *DropD-SLAM*. Please make sure you have installed all required dependencies (see section 2).

```bash
chmod +x build.sh
./build.sh
```

This will create **libDropD_SLAM.so** at *lib* folder and the executables in *Examples* folder.

## Verify Installation

After building, verify that all models are in place:

```bash
ls -lh models/
# Should show:
#   - unidepthv2_with_cam.onnx
#   - keynet.onnx
#   - yolo11s-seg.onnx (optional, for dynamic scenes)
```

# 4. Running DropD-SLAM

## Quick Start with TUM RGB-D Dataset

Download a sequence from the [TUM RGB-D benchmark](https://vision.in.tum.de/data/datasets/rgbd-dataset):

```bash
# Example: Download freiburg1_desk2
cd ~
wget https://vision.in.tum.de/rgbdslam/data/rgbd_dataset_freiburg1_desk2.tgz
tar -xzf rgbd_dataset_freiburg1_desk2.tgz
cd /path/to/DropD-SLAM
```

Run DropD-SLAM with monocular depth estimation:

```bash
./Examples/RGB-D/rgbd_tum \
    Vocabulary/ORBvoc.txt \
    Examples/RGB-D/TUM1.yaml \
    ~/rgbd_dataset_freiburg1_desk2 \
    ~/rgbd_dataset_freiburg1_desk2/associations_sensor.txt \
    unidepth \
    models/unidepthv2_with_cam.onnx
```

### Command Line Arguments

```
./Examples/RGB-D/rgbd_tum <vocabulary> <config> <sequence_path> <associations> <depth_model_type> <depth_model_path>
```

- `<vocabulary>`: Path to ORB vocabulary file
- `<config>`: Path to YAML configuration file
- `<sequence_path>`: Path to TUM RGB-D dataset sequence
- `<associations>`: Path to associations file
- `<depth_model_type>`: Type of depth model (`unidepth` or `depthanything`)
- `<depth_model_path>`: Path to ONNX depth model

## Configuration

The configuration file (`Examples/RGB-D/TUM1.yaml`) contains important settings:

### Dynamic Scene Detection

```yaml
# Enable dynamic object detection and masking (requires YoloSegmentator)
# Set to 0 to disable, 1 to enable
System.DynamicScene: 0  # Recommended: 0 for static scenes, 1 for dynamic scenes
```

- **0 (disabled)**: Faster, lower memory usage, suitable for static scenes
- **1 (enabled)**: Robust to dynamic objects, higher memory usage

### Camera Parameters

Ensure your camera intrinsics are correctly calibrated in the YAML file:

```yaml
Camera1.fx: 517.306408
Camera1.fy: 516.469215
Camera1.cx: 318.643040
Camera1.cy: 255.313989
```

## Performance Tips

1. **Dynamic Scenes**: Enable `System.DynamicScene: 1` only when needed

## Running time analysis
A flag in `include\Config.h` activates time measurements. It is necessary to uncomment the line `#define REGISTER_TIMES` to obtain the time stats of one execution which is shown at the terminal and stored in a text file(`ExecTimeMean.txt`).


# 5. Acknowledgments

DropD-SLAM integrates several state-of-the-art deep learning models:

## Monocular Depth Estimation
- **UniDepth** ([GitHub](https://github.com/lpiccinelli-eth/UniDepth)) - Metric monocular depth estimation
  - Piccinelli, L., et al. "UniDepth: Universal Monocular Metric Depth Estimation"

## Keypoint Detection and Description  
- **KeyNet** ([GitHub](https://github.com/axelBarroso/Key.Net)) - Learned keypoint detector
  - Barroso-Laguna, A., et al. "Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters"
- **HyNet** - Learned descriptor network

## Instance Segmentation
- **YOLO11** ([Ultralytics](https://github.com/ultralytics/ultralytics)) - Real-time object detection and segmentation
  - Jocher, G., et al. "Ultralytics YOLO11"

## Backend SLAM System
- **ORB-SLAM3** ([GitHub](https://github.com/UZ-SLAMLab/ORB_SLAM3)) - Visual-Inertial SLAM
  - Campos, C., et al. "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM"

We thank the authors of these excellent works for making their code and models publicly available.

# 6. Troubleshooting

## CUDA/GPU Issues

### "CUDA not available" or "CUDA out of memory"
```bash
# Check CUDA installation
nvidia-smi

# Check ONNX Runtime GPU support
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should include 'CUDAExecutionProvider'

# Reduce memory usage by disabling dynamic scene detection
# In your YAML config: System.DynamicScene: 0
```

### "libonnxruntime.so: cannot open shared object file"
```bash
# Add ONNX Runtime to library path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

## Model Loading Issues

### "Failed to load model" or "Model not found"
```bash
# Verify all models are downloaded and in correct location
ls -lh models/
# Should show .onnx files with sizes > 100MB

# Check file permissions
chmod 644 models/*.onnx
```

### "Type mismatch" or "Shape mismatch" errors
- Ensure you're using the correct model export (with or without --with-camera flag)
- Verify input dimensions match model expectations (476x630 for UniDepth)

## Performance Issues

### "Slow inference time" (>200ms per frame)
1. Verify GPU is being used:
   ```bash
   # While running, check GPU usage
   watch -n 1 nvidia-smi
   ```
2. Check CUDA provider is loaded (not falling back to CPU)
3. Ensure TensorRT cache is being used (first run builds cache, subsequent runs are faster)
4. Consider reducing input resolution in model export

### "Tracking fails" or "Lost tracking"
1. **Disable dynamic scene detection** if scene is static: `System.DynamicScene: 0`
2. Check depth map quality - visualize intermediate results
3. Verify camera intrinsics are correctly calibrated
4. Ensure sufficient lighting and texture in the scene
5. Try different depth model (with/without camera intrinsics)

## Build Issues

### "onnxruntime_cxx_api.h: No such file or directory"
```bash
# Reinstall ONNX Runtime headers
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-gpu-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.16.3.tgz
sudo cp -r onnxruntime-linux-x64-gpu-1.16.3/include/* /usr/local/include/
```

### Undefined reference to ONNX Runtime functions
```bash
# Check library is linked correctly
sudo cp -r onnxruntime-linux-x64-gpu-1.16.3/lib/* /usr/local/lib/
sudo ldconfig
```

## Getting Help

For issues specific to DropD-SLAM:
- Check existing GitHub issues
- Provide full error logs and system information
- Include GPU model, CUDA version, and ONNX Runtime version

For issues with ORB-SLAM3 backend:
- Refer to [ORB-SLAM3 documentation](https://github.com/UZ-SLAMLab/ORB_SLAM3)

For issues with specific models:
- UniDepth: [GitHub Issues](https://github.com/lpiccinelli-eth/UniDepth/issues)
- KeyNet: [GitHub Issues](https://github.com/axelBarroso/Key.Net/issues)  
- YOLO: [Ultralytics Docs](https://docs.ultralytics.com/)

# 7. TODO(WIP)

The following items are planned for future releases:

1. **Add DepthAnythingV2 Support**
   - Full integration  of DepthAnythingV2 depth estimation model
   - Configuration options for model selection

2. **ONNX Model Export Guidelines**
   - Comprehensive documentation for exporting custom depth models to ONNX format
   - Step-by-step guide for model conversion from PyTorch/TensorFlow
   - Best practices for model optimization and quantization
