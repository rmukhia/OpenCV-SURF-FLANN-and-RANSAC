cmake_minimum_required(VERSION 3.10)
project(hw3)
find_package(OpenCV)
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

include_directories("/opt/local/include")
include_directories( ${OpenCV_INCLUDE_DIRS} )

link_directories("/opt/local/lib")
link_directories("/opt/local/opencv-3.4.1/lib")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -g -Dsprintf_s=sprintf")

add_executable(hw3 main.cpp)

# target_link_libraries(hw3 opencv_core opencv_highgui opencv_imgcodecs opencv_features2d opencv_xfeatures2d)
target_link_libraries(hw3 ${OpenCV_LIBS})