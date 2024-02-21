#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <tuple>

#include "line_detector.h"

cv::Mat filter_image(const cv::Mat &frame) {
    cv::Mat hls_frame;
    cv::cvtColor(frame, hls_frame, cv::COLOR_BGR2HLS);

    std::vector<int> upper_bound_white{255, 255, 255};
    std::vector<int> lower_bound_white{0, 150, 10};

    cv::Mat white_mask;
    cv::Mat yellow_mask;
    cv::Mat full_mask;
    cv::inRange(hls_frame, lower_bound_white, upper_bound_white, white_mask);

    cv::Mat filtered_hls_frame;
    cv::bitwise_and(hls_frame, hls_frame, filtered_hls_frame, white_mask);

    cv::Mat filtered_rgb_frame;
    cv::cvtColor(filtered_hls_frame, filtered_rgb_frame, cv::COLOR_HLS2BGR);

    return filtered_rgb_frame;
}

cv::Mat mask_image(const cv::Mat &frame) {
    cv::Mat mask = cv::Mat(frame.rows, frame.cols, CV_8U, 1);
    mask(cv::Rect(0, 0, frame.cols, round(frame.rows * 0.3))) = 0;

    cv::Mat masked_image;
    cv::bitwise_and(frame, frame, masked_image, mask);

    return masked_image;
}

cv::Mat edge_detection(const cv::Mat &frame) {
    cv::Mat grayscale_image;
    cv::cvtColor(frame, grayscale_image, cv::COLOR_BGR2GRAY);

    cv::Mat denoised_image;
    cv::medianBlur(grayscale_image, denoised_image, 5);

    cv::Mat edge_image;
    cv::Canny(denoised_image, edge_image, 50, 150);

    return edge_image;
}

void display_process(const cv::Mat &base_image, const cv::Mat &filtered_image, const cv::Mat &masked_image, const cv::Mat &line_image) {
    cv::Mat resized_base_image;
    cv::resize(base_image, resized_base_image, cv::Size(960, 540));
    
    cv::Mat resized_filtered_image;
    cv::resize(filtered_image, resized_filtered_image, cv::Size(960, 540));

    cv::Mat resized_masked_image;
    cv::resize(masked_image, resized_masked_image, cv::Size(960, 540));

    cv::Mat color_edge_image;
    cv::cvtColor(line_image, color_edge_image, cv::COLOR_GRAY2BGR);

    cv::Mat resized_line_image;
    cv::resize(color_edge_image, resized_line_image, cv::Size(960, 540));

    cv::Mat top_images;
    cv::Mat bot_images;
    cv::Mat all_images;
    cv::hconcat(resized_base_image, resized_filtered_image, top_images);
    cv::hconcat(resized_masked_image, resized_line_image, bot_images);
    cv::vconcat(top_images, bot_images, all_images);

    cv::imshow("process", all_images);
    cv::waitKey(25);
}

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

    line_detection::LineDetector line_detector;
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    while (true) {
        cap.read(frame);

        if (frame.empty()) {
            std::cerr << "ERROR! opened frame is empty \n";
        }

        cv::Mat filtered_image = filter_image(frame);
        cv::Mat masked_image = mask_image(filtered_image);
        cv::Mat edge_image = edge_detection(masked_image);

        auto [left_line, right_line] = line_detector.find_lines(edge_image);

        display_process(frame, filtered_image, masked_image, edge_image);
        display_result(frame, left_line, right_line);
        

    }

    return 0;
}