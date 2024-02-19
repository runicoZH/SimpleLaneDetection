#include <iostream>
#include <numeric>
#include <tuple>

#include <opencv2/opencv.hpp>

#include "line_detector.h"


cv::Mat filter_image(const cv::Mat &frame){

    cv::Mat hls_frame;
    cv::cvtColor(frame, hls_frame, cv::COLOR_BGR2HLS);
    
    std::vector<int> upper_bound_white {255, 255, 100};
    std::vector<int> lower_bound_white {0, 150, 0};

    std::vector<int> upper_bound_yellow {50, 255, 255};
    std::vector<int> lower_bound_yellow {20, 50, 90};

    cv::Mat white_mask;
    cv::Mat yellow_mask;
    cv::Mat full_mask;
    cv::inRange(hls_frame, lower_bound_white, upper_bound_white, white_mask);
    cv::inRange(hls_frame, lower_bound_yellow, upper_bound_yellow, yellow_mask);
    cv::bitwise_or(white_mask, yellow_mask, full_mask);

    cv::Mat filtered_hls_frame;
    cv::bitwise_and(hls_frame, hls_frame, filtered_hls_frame, full_mask);

    cv::Mat filtered_rgb_frame; 
    cv::cvtColor(filtered_hls_frame, filtered_rgb_frame, cv::COLOR_HLS2BGR);

    return filtered_rgb_frame;
}

cv::Mat mask_image(const cv::Mat &frame){

    cv::Mat mask = cv::Mat(frame.rows, frame.cols, CV_8U, 1);
    mask(cv::Rect(0,0,frame.cols, round(frame.rows*0.3))) = 0;
    
    cv::Mat masked_image;
    cv::bitwise_and(frame, frame, masked_image, mask);

    return masked_image;
}

cv::Mat edge_detection(const cv::Mat &frame){
    
    cv::Mat grayscale_image;
    cv::cvtColor(frame, grayscale_image, cv::COLOR_BGR2GRAY);

    cv::Mat edge_image;
    cv::Canny(grayscale_image, edge_image, 50, 150);

    return edge_image;
}

void display(const cv::Mat &frame){

    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(920, 540));
    cv::imshow("display", resized_frame);
    cv::waitKey(25);
}

int main(int argc, char *argv[])
{

    cv::Mat frame;
    cv::VideoCapture cap("/home/nico/workspace/LaneDetection/resources/dashcam.mp4");

    line_detection::LineDetector line_detector;
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    while (true){
        cap.read(frame);

        if (frame.empty()){
            std::cerr << "ERROR! opened frame is empty \n";
        }

        cv::Mat filtered_image = filter_image(frame);
        cv::Mat masked_image = mask_image(filtered_image);
        cv::Mat edge_image = edge_detection(masked_image);
        auto [left_line, right_line] = line_detector.find_lines(edge_image);

        cv::line( frame, cv::Point(left_line[0], left_line[1]), cv::Point(left_line[2], left_line[3]), (0,255,0), 10, cv::LINE_AA);
        cv::line( frame, cv::Point(right_line[0], right_line[1]), cv::Point(right_line[2], right_line[3]), (0,0,255), 10, cv::LINE_AA);
        display(frame);
    }

    return 0;
}