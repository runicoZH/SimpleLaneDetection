cmake_minimum_required(VERSION 3.0)
project( LaneDetection )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES main.cpp line_detector.cpp)

add_executable( ${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries( LaneDetection ${OpenCV_LIBS} )