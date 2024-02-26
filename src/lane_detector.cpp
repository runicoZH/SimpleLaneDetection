
#include "lane_detector.h"

#include "math.h"
namespace lane_detection {

std::tuple<cv::Vec4i, cv::Vec4i> LaneDetector::find_lanes(
    const cv::Mat &frame) {
    cv::Mat filtered_image = filter_image_(frame);
    cv::Mat masked_image = mask_image_(filtered_image);
    cv::Mat edge_image = edge_detection_(masked_image);
    display_process_(frame, filtered_image, masked_image, edge_image);

    auto [left_ret, left_slope, left_intercept, right_ret, right_slope,
          right_intercept] = line_detection_(edge_image);

    float left_avg_slope;
    int left_avg_intercept;
    if (left_ret) {
        left_avg_slope = moving_average_slopes_(past_left_slopes_, left_slope,
                                                sum_left_slopes_);
        left_avg_intercept = moving_average_intercepts_(
            past_left_intercepts_, left_intercept, sum_left_intercepts_);
    } else {
        left_avg_slope = sum_left_slopes_ / int(past_left_slopes_.size());
        left_avg_intercept =
            sum_left_intercepts_ / int(past_left_intercepts_.size());
    }

    int x1_left = left_avg_intercept - (width_ / left_avg_slope);
    cv::Vec4i left_line = cv::Vec4i(x1_left, 0, left_avg_intercept, width_);

    float right_avg_slope;
    int right_avg_intercept;
    if (right_ret) {
        right_avg_slope = moving_average_slopes_(
            past_right_slopes_, right_slope, sum_right_slopes_);
        right_avg_intercept = moving_average_intercepts_(
            past_right_intercepts_, right_intercept, sum_right_intercepts_);
    } else {
        right_avg_slope = sum_right_slopes_ / int(past_right_slopes_.size());
        right_avg_intercept =
            sum_right_intercepts_ / int(past_right_intercepts_.size());
    }

    int x1_right = right_avg_intercept - (width_ / right_avg_slope);
    cv::Vec4i right_line = cv::Vec4i(x1_right, 0, right_avg_intercept, width_);

    return {left_line, right_line};
}

int LaneDetector::moving_average_intercepts_(std::queue<int> &past_intercepts,
                                             int new_intercept,
                                             int &current_sum) {
    if (past_intercepts.size() > 10) {
        current_sum -= past_intercepts.front();
        past_intercepts.pop();
    }

    past_intercepts.push(new_intercept);
    current_sum += new_intercept;

    return current_sum / int(past_intercepts.size());
}

float LaneDetector::moving_average_slopes_(std::queue<float> &past_slopes,
                                           float new_slope,
                                           float &current_sum) {
    if (past_slopes.size() > 10) {
        current_sum -= past_slopes.front();
        past_slopes.pop();
    }

    past_slopes.push(new_slope);
    current_sum += new_slope;

    return current_sum / float(past_slopes.size());
}

cv::Mat LaneDetector::filter_image_(const cv::Mat &frame) {
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

cv::Mat LaneDetector::mask_image_(const cv::Mat &frame) {
    cv::Mat mask = cv::Mat(width_, height_, CV_8U, 1);
    mask(cv::Rect(0, 0, height_, round(width_ * 0.3))) = 0;

    cv::Mat masked_image;
    cv::bitwise_and(frame, frame, masked_image, mask);

    return masked_image;
}

cv::Mat LaneDetector::edge_detection_(const cv::Mat &frame) {
    cv::Mat grayscale_image;
    cv::cvtColor(frame, grayscale_image, cv::COLOR_BGR2GRAY);

    cv::Mat denoised_image;
    cv::medianBlur(grayscale_image, denoised_image, 5);

    cv::Mat edge_image;
    cv::Canny(denoised_image, edge_image, 50, 150);

    return edge_image;
}

std::tuple<bool, float, int, bool, float, int> LaneDetector::line_detection_(
    const cv::Mat &frame) {
    std::vector<cv::Vec4i> output_lines;
    cv::HoughLinesP(frame, output_lines, 1, CV_PI / 180, 50, 60, 10);

    std::vector<float> left_slopes;
    std::vector<int> left_intercepts;
    std::vector<float> right_slopes;
    std::vector<int> right_intercepts;
    for (size_t i = 0; i < output_lines.size(); i++) {
        cv::Vec4i l = output_lines[i];
        if (l[3] - l[1] == 0 || l[2] - l[0] == 0) {
            continue;
        }
        float slope = float(l[3] - l[1]) / float(l[2] - l[0]);
        int intercept = (width_ - l[1] + (slope * l[0])) / slope;
        if (slope < -0.5 && intercept < height_ / 2) {
            left_slopes.push_back(slope);
            left_intercepts.push_back(intercept);
        }

        if (slope > 0.5 && intercept > height_ / 2) {
            right_slopes.push_back(slope);
            right_intercepts.push_back(intercept);
        }
    }

    float left_avg_slope;
    int left_avg_intercept;
    bool left_ret = false;
    if (left_slopes.size() > 0) {
        left_ret = true;
        left_avg_slope = std::reduce(left_slopes.begin(), left_slopes.end()) /
                         float(left_slopes.size());
        left_avg_intercept =
            std::reduce(left_intercepts.begin(), left_intercepts.end()) /
            int(left_intercepts.size());
    }

    bool right_ret = false;
    float right_avg_slope;
    int right_avg_intercept;
    if (right_slopes.size() > 0) {
        right_ret = true;
        right_avg_slope =
            std::reduce(right_slopes.begin(), right_slopes.end()) /
            float(right_slopes.size());
        right_avg_intercept =
            std::reduce(right_intercepts.begin(), right_intercepts.end()) /
            int(right_intercepts.size());
    }

    return {left_ret,  left_avg_slope,  left_avg_intercept,
            right_ret, right_avg_slope, right_avg_intercept};
}

void LaneDetector::display_process_(const cv::Mat &base_image,
                                    const cv::Mat &filtered_image,
                                    const cv::Mat &masked_image,
                                    const cv::Mat &line_image) {
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

}  // namespace lane_detection