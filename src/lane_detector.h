#ifndef LINE_DETECTOR_H
#define LINE_DETECTOR_H

#include <numeric>
#include <opencv2/opencv.hpp>
#include <queue>
#include <tuple>
#include <vector>

namespace lane_detection {

class LaneDetector {
   public:
    LaneDetector(){};

    std::tuple<cv::Vec4i, cv::Vec4i> find_lanes(const cv::Mat &frame);

   private:
    std::queue<int> past_left_intercepts_;
    int sum_left_intercepts_ = 0;
    std::queue<float> past_left_slopes_;
    float sum_left_slopes_ = 0;

    std::queue<int> past_right_intercepts_;
    int sum_right_intercepts_ = 0;
    std::queue<float> past_right_slopes_;
    float sum_right_slopes_ = 0;

    int moving_average_intercepts_(std::queue<int> &past_intercepts,
                                   int new_intercept, int &current_sum);
    float moving_average_slopes_(std::queue<float> &past_slopes,
                                 float new_slope, float &current_sum);

    cv::Mat filter_image(const cv::Mat &frame);

    cv::Mat mask_image(const cv::Mat &frame);

    cv::Mat edge_detection(const cv::Mat &frame);

    void display_process(const cv::Mat &base_image,
                         const cv::Mat &filtered_image,
                         const cv::Mat &masked_image,
                         const cv::Mat &line_image);
};

}  // namespace lane_detection

#endif