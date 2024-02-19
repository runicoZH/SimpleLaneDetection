
#include "line_detector.h"

#include "math.h"
namespace line_detection{

int LineDetector::moving_average_intercepts_(std::queue<int> &past_intercepts, int new_intercept, int &current_sum){

    if (past_intercepts.size() > 10){
        current_sum -= past_intercepts.front();
        past_intercepts.pop();
    }

    past_intercepts.push(new_intercept);
    current_sum += new_intercept;

    return current_sum/int(past_intercepts.size());
}

float LineDetector::moving_average_slopes_(std::queue<float> &past_slopes, float new_slope, float &current_sum){

    if (past_slopes.size() > 10){
        current_sum -= past_slopes.front();
        past_slopes.pop();
    }

    past_slopes.push(new_slope);
    current_sum += new_slope;

    return current_sum/float(past_slopes.size());
}

std::tuple<cv::Vec4i, cv::Vec4i> LineDetector::find_lines(const cv::Mat &frame){
    
    std::vector<cv::Vec4i>  output_lines;
    cv::HoughLinesP(frame, output_lines, 1, CV_PI/180, 50, 60, 10);

    std::vector<float>  left_slopes;
    std::vector<int>  left_intercepts;
    std::vector<float>  right_slopes;
    std::vector<int>  right_intercepts;
    for( size_t i = 0; i < output_lines.size(); i++ )
        {
            cv::Vec4i l = output_lines[i];
            if (l[3]-l[1] == 0 || l[2]-l[0] == 0){
                continue;
            }
            float slope = float(l[3]-l[1]) / float(l[2]-l[0]);
            int intercept = (frame.rows - l[1]+ (slope*l[0]))/slope;
            if (slope < -0.5 && intercept < frame.cols/2){
                left_slopes.push_back(slope);
                left_intercepts.push_back(intercept);
            }

            if (slope > 0.5 && intercept > frame.cols/2){
                right_slopes.push_back(slope);
                right_intercepts.push_back(intercept);
            }

        }
    int x1;
    cv::Vec4i left_line;
    if (left_slopes.size() > 0){
        float left_avg_slope = std::reduce(left_slopes.begin(), left_slopes.end())/float(left_slopes.size());
        int left_avg_intercept = std::reduce(left_intercepts.begin(), left_intercepts.end())/int(left_intercepts.size());
        
        left_avg_slope = moving_average_slopes_(past_left_slopes_, left_avg_slope, sum_left_slopes_);
        left_avg_intercept = moving_average_intercepts_(past_left_intercepts_, left_avg_intercept, sum_left_intercepts_);
        
        x1 = left_avg_intercept - (frame.rows/left_avg_slope);
        left_line = cv::Vec4i(x1, 0, left_avg_intercept, frame.rows);
    }

    cv::Vec4i right_line;
    if (right_slopes.size() > 0){
        float right_avg_slope = std::reduce(right_slopes.begin(), right_slopes.end())/float(right_slopes.size());
        int right_avg_intercept = std::reduce(right_intercepts.begin(), right_intercepts.end())/int(right_intercepts.size());
        
        right_avg_slope = moving_average_slopes_(past_right_slopes_, right_avg_slope, sum_right_slopes_);
        right_avg_intercept = moving_average_intercepts_(past_right_intercepts_, right_avg_intercept, sum_right_intercepts_);

        x1 = right_avg_intercept - (frame.rows/right_avg_slope);
        right_line = cv::Vec4i(x1, 0, right_avg_intercept, frame.rows);
    }

    return {left_line, right_line};
}

} // namespace line_detection