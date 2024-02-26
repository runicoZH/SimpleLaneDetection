#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <tuple>

#include <lane_detector.h>


void display_result(const cv::Mat &base_image, cv::Vec4i left_line, cv::Vec4i right_line){

    cv::Mat result_image = base_image;

    cv::line(result_image, cv::Point(left_line[0], left_line[1]),
                 cv::Point(left_line[2], left_line[3]), (0, 255, 0), 10,
                 cv::LINE_AA);
    cv::line(result_image, cv::Point(right_line[0], right_line[1]),
                cv::Point(right_line[2], right_line[3]), (0, 0, 255), 10,
                cv::LINE_AA);


    cv::Mat resized_result_image;
    cv::resize(result_image, resized_result_image, cv::Size(960, 540));

    cv::imshow("Result", resized_result_image);
    cv::waitKey(25);
}


int main(int argc, char *argv[]) {
    cv::Mat frame;
    cv::VideoCapture cap(
        "/home/nico/workspace/LaneDetection/resources/dashcam.mp4");

    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    cap.read(frame);
    if (frame.empty()) {
        std::cerr << "ERROR! opened frame is empty \n";
    }

    lane_detection::LaneDetector lane_detector(frame.rows, frame.cols);

    while (true) {
        cap.read(frame);

        if (frame.empty()) {
            std::cerr << "ERROR! opened frame is empty \n";
        }

        auto [left_line, right_line] = lane_detector.find_lanes(frame);
        display_result(frame, left_line, right_line);
        

    }

    return 0;
}