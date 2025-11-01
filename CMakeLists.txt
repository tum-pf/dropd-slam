cmake_minimum_required(VERSION 2.8)
project(DropD_SLAM LANGUAGES CXX CUDA)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}    -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# Enforce modern C++ across the project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
add_definitions(-DCOMPILEDWITHC17 -DCOMPILEDWITHC14)
message(STATUS "Using flag -std=c++17.")

# Ensure OpenGL extension function prototypes are declared when including legacy headers
add_definitions(-DGL_GLEXT_PROTOTYPES)

set(ONNXRUNTIME_ROOT "/opt/onnxruntime-linux-x64-gpu-1.22.0")
set(ONNXRUNTIME_LIB_DIRS "/opt/onnxruntime-linux-x64-gpu-1.22.0/lib")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${ONNXRUNTIME_ROOT})
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT}/include)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


# Find CUDA explicitly
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "89")

# CUDA-specific flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -lineinfo")

# Include CUDA headers
include_directories(${CUDA_INCLUDE_DIRS})


find_package(OpenCV 4.2)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV >= 4.2 not found.")
   endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(onnxruntime)

# Existing includes
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/CameraModels
  ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
file(GLOB CUDA_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cu)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Keynet.cpp
src/Atlas.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Viewer.cc
src/ImuTypes.cc
src/G2oTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/MLPnPsolver.cpp
src/GeometricTools.cc
src/TwoViewReconstruction.cc
src/Config.cc
src/Settings.cc
src/YoloSegmentator.cpp
src/DepthInference.cpp
include/System.h
include/Tracking.h
include/LocalMapping.h
include/LoopClosing.h
include/ORBextractor.h
include/ORBmatcher.h
include/FrameDrawer.h
include/Converter.h
include/MapPoint.h
include/KeyFrame.h
include/Keynet.hpp
include/Atlas.h
include/Map.h
include/MapDrawer.h
include/Optimizer.h
include/Frame.h
include/KeyFrameDatabase.h
include/Sim3Solver.h
include/Viewer.h
include/ImuTypes.h
include/G2oTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/OptimizableTypes.h
include/MLPnPsolver.h
include/GeometricTools.h
include/TwoViewReconstruction.h
include/SerializationUtils.h
include/Config.h
include/Settings.h
include/YoloSegmentator.hpp
include/DepthInference.h
${CUDA_SOURCES}
)

add_subdirectory(Thirdparty/g2o)


# Link CUDA libraries explicitly
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME}
   PUBLIC
      ${OpenCV_LIBS}
      ${EIGEN3_LIBS}
      ${Pangolin_LIBRARIES}
      ${ONNXRUNTIME_LIB_DIRS}/libonnxruntime.so.1.22.0
      ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
      ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
      ${CUDA_LIBRARIES}
      -lboost_serialization
      -lcrypto
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Enable OpenMP if available for parallel loops
find_package(OpenMP)
if(OPENMP_FOUND)
   message(STATUS "OpenMP found: enabling OpenMP flags")
   target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
   if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(${PROJECT_NAME} PUBLIC -fopenmp)
   endif()
endif()


# Build examples

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})
target_include_directories(rgbd_tum PRIVATE ${OpenCV_INCLUDE_DIRS})

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
        Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})
target_include_directories(mono_tum PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(mono_kitti
        Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})
target_include_directories(mono_kitti PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(mono_euroc
        Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})
target_include_directories(mono_euroc PRIVATE ${OpenCV_INCLUDE_DIRS})

#Old examples

# # RGB-D examples
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/RGB-D)

# add_executable(rgbd_tum_old
#         Examples_old/RGB-D/rgbd_tum.cc)
# target_link_libraries(rgbd_tum_old ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(rgbd_realsense_D435i_old
#             Examples_old/RGB-D/rgbd_realsense_D435i.cc)
#     target_link_libraries(rgbd_realsense_D435i_old ${PROJECT_NAME})
# endif()


# # RGB-D inertial examples
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/RGB-D-Inertial)

# if(realsense2_FOUND)
#     add_executable(rgbd_inertial_realsense_D435i_old
#             Examples_old/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
#     target_link_libraries(rgbd_inertial_realsense_D435i_old ${PROJECT_NAME})
# endif()

# #Stereo examples
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Stereo)

# add_executable(stereo_kitti_old
#         Examples_old/Stereo/stereo_kitti.cc)
# target_link_libraries(stereo_kitti_old ${PROJECT_NAME})

# add_executable(stereo_euroc_old
#         Examples_old/Stereo/stereo_euroc.cc)
# target_link_libraries(stereo_euroc_old ${PROJECT_NAME})

# add_executable(stereo_tum_vi_old
#         Examples_old/Stereo/stereo_tum_vi.cc)
# target_link_libraries(stereo_tum_vi_old ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(stereo_realsense_t265_old
#             Examples_old/Stereo/stereo_realsense_t265.cc)
#     target_link_libraries(stereo_realsense_t265_old ${PROJECT_NAME})

#     add_executable(stereo_realsense_D435i_old
#             Examples_old/Stereo/stereo_realsense_D435i.cc)
#     target_link_libraries(stereo_realsense_D435i_old ${PROJECT_NAME})
# endif()

# #Monocular examples
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Monocular)

# add_executable(mono_tum_old
#         Examples_old/Monocular/mono_tum.cc)
# target_link_libraries(mono_tum_old ${PROJECT_NAME})

# add_executable(mono_kitti_old
#         Examples_old/Monocular/mono_kitti.cc)
# target_link_libraries(mono_kitti_old ${PROJECT_NAME})

# add_executable(mono_euroc_old
#         Examples_old/Monocular/mono_euroc.cc)
# target_link_libraries(mono_euroc_old ${PROJECT_NAME})

# add_executable(mono_tum_vi_old
#         Examples_old/Monocular/mono_tum_vi.cc)
# target_link_libraries(mono_tum_vi_old ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(mono_realsense_t265_old
#             Examples_old/Monocular/mono_realsense_t265.cc)
#     target_link_libraries(mono_realsense_t265_old ${PROJECT_NAME})

#     add_executable(mono_realsense_D435i_old
#             Examples_old/Monocular/mono_realsense_D435i.cc)
#     target_link_libraries(mono_realsense_D435i_old ${PROJECT_NAME})
# endif()

# #Monocular inertial examples
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Monocular-Inertial)

# add_executable(mono_inertial_euroc_old
#         Examples_old/Monocular-Inertial/mono_inertial_euroc.cc)
# target_link_libraries(mono_inertial_euroc_old ${PROJECT_NAME})

# add_executable(mono_inertial_tum_vi_old
#         Examples_old/Monocular-Inertial/mono_inertial_tum_vi.cc)
# target_link_libraries(mono_inertial_tum_vi_old ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(mono_inertial_realsense_t265_old
#             Examples_old/Monocular-Inertial/mono_inertial_realsense_t265.cc)
#     target_link_libraries(mono_inertial_realsense_t265_old ${PROJECT_NAME})

#     add_executable(mono_inertial_realsense_D435i_old
#             Examples_old/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
#     target_link_libraries(mono_inertial_realsense_D435i_old ${PROJECT_NAME})
# endif()

# #Stereo Inertial examples
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Stereo-Inertial)

# add_executable(stereo_inertial_euroc_old
#         Examples_old/Stereo-Inertial/stereo_inertial_euroc.cc)
# target_link_libraries(stereo_inertial_euroc_old ${PROJECT_NAME})

# add_executable(stereo_inertial_tum_vi_old
#         Examples_old/Stereo-Inertial/stereo_inertial_tum_vi.cc)
# target_link_libraries(stereo_inertial_tum_vi_old ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(stereo_inertial_realsense_t265_old
#             Examples_old/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
#     target_link_libraries(stereo_inertial_realsense_t265_old ${PROJECT_NAME})

#     add_executable(stereo_inertial_realsense_D435i_old
#             Examples_old/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
#     target_link_libraries(stereo_inertial_realsense_D435i_old ${PROJECT_NAME})
# endif()
