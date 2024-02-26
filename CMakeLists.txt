cmake_minimum_required(VERSION 3.16)
project( LaneDetection )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_subdirectory(src)

add_subdirectory(apps)