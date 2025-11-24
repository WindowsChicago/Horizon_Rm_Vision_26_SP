cmake_minimum_required(VERSION 3.16)

set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# Add module in charge of finding TensorRT
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

add_subdirectory(planner/tinympc)

add_library(auto_aim OBJECT 
    armor.cpp
    classifier.cpp 
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov5_trt.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
    trt_engine.cpp
    int8_calibrator.cpp
)

target_link_libraries(auto_aim PUBLIC ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} io openvino::runtime tinympcstatic )
target_include_directories(auto_aim PUBLIC include ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})