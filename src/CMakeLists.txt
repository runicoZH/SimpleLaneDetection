

add_library(LaneDetector line_detector.cpp line_detector.h)

target_link_libraries(LaneDetector PRIVATE ${OpenCV_LIBS})
target_include_directories(LaneDetector PUBLIC ../src)
