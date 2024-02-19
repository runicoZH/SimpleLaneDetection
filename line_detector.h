# ifndef LINE_DETECTOR_H
# define LINE_DETECTOR_H

#include <numeric>
#include <tuple>
#include <vector>
#include <queue>

#include <opencv2/opencv.hpp>

namespace line_detection{

class LineDetector {

public:
    LineDetector(){};

    std::tuple<cv::Vec4i, cv::Vec4i> find_lines(const cv::Mat &frame);

private:
    std::queue<int> past_left_intercepts_;
    int sum_left_intercepts_ = 0;
    std::queue<float> past_left_slopes_;
    float sum_left_slopes_ = 0;
    
    std::queue<int> past_right_intercepts_;
    int sum_right_intercepts_ = 0;
    std::queue<float> past_right_slopes_;
    float sum_right_slopes_ = 0;

    int moving_average_intercepts_(std::queue<int> &past_intercepts, int new_intercept, int &current_sum);
    float moving_average_slopes_(std::queue<float> &past_slopes, float new_slope, float &current_sum);
};




} // namespace line_detection

#endif