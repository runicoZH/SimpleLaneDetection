
add_executable(DetectLanes main.cpp)

target_link_libraries(DetectLanes PRIVATE LaneDetector ${OpenCV_LIBS})