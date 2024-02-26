

add_library(LaneDetector lane_detector.cpp lane_detector.h)

target_link_libraries(LaneDetector PRIVATE ${OpenCV_LIBS})
target_include_directories(LaneDetector PUBLIC ../src)
