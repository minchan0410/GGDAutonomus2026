#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Int16.h>
#include <std_msgs/Int32MultiArray.h>
#include <geometry_msgs/PointStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <deque>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <thread>
#include <mutex>

using namespace cv;
using namespace std;

// ==========================================
// 설정 변수
// ==========================================
const string IMAGE_TOPIC = "/cam1/usb_cam/image_raw"; 

class LaneDetector {
private:
    ros::NodeHandle nh;
    ros::NodeHandle pnh;
    ros::Publisher pub_steer;
    ros::Publisher pub_lines_px;
    ros::Publisher pub_target_px;
    ros::Publisher pub_cross;
    ros::Subscriber image_sub;

    // 이미지 메모리 재사용
    Mat gray_img, blur_img, edge_img, roi_mask;
    Mat kernel_dilate; // [NEW] 팽창 연산용 커널

    // 조향 및 크로스라인 히스토리
    deque<int> steer_history;
    const int window_size = 15;
    
    deque<int> cross_queue;
    const int cross_queue_maxlen = 10;
    const int cross_threshold = 20000;

    // 시각화 스레드 관련
    Mat display_img;
    mutex display_mutex;
    bool new_display_data = false;
    bool is_running = true;
    thread vis_thread;

public:
    LaneDetector() : pnh("~") {
        string output_topic_name;
        pnh.param<string>("output_topic", output_topic_name, "des_steer");
        
        pub_steer = nh.advertise<std_msgs::Int16>(output_topic_name, 10);
        pub_lines_px = nh.advertise<std_msgs::Int32MultiArray>("/lane_lines_px", 10);
        pub_target_px = nh.advertise<geometry_msgs::PointStamped>("/lane_target_px", 10);
        pub_cross = nh.advertise<std_msgs::Int16>("/crossline", 10);

        image_sub = nh.subscribe(IMAGE_TOPIC, 1, &LaneDetector::imageCallback, this);

        // [NEW] 팽창용 커널 미리 생성 (3x3 사각형)
        kernel_dilate = getStructuringElement(MORPH_RECT, Size(3, 3));

        ROS_INFO("Lane Detector Node (CPU) Started.");
        vis_thread = thread(&LaneDetector::displayWorker, this);
    }

    ~LaneDetector() {
        is_running = false;
        if (vis_thread.joinable()) vis_thread.join();
        destroyAllWindows();
    }

    void displayWorker() {
        while (is_running && ros::ok()) {
            Mat img_to_show;
            {
                lock_guard<mutex> lock(display_mutex);
                if (new_display_data) {
                    display_img.copyTo(img_to_show);
                    new_display_data = false;
                }
            }
            if (!img_to_show.empty()) {
                imshow("Lane Detector (CUDA)", img_to_show);
                waitKey(1);
            } else {
                this_thread::sleep_for(chrono::milliseconds(10));
            }
        }
    }

    // ==========================================
    // 보조 함수
    // ==========================================
    float calculate_midpoint_score(int x1, int x2, int w) {
        float hw = w / 2.0f;
        int midpoint = (x1 + x2) / 2;
        return (midpoint - hw) / hw * 100.0f;
    }

    float calculate_line_score(int x1, int y1, int x2, int y2) {
        if (x2 - x1 == 0) return 0.0f;
        float slope = (float)(y2 - y1) / (x2 - x1);
        float abs_slope = abs(slope);
        
        if (abs_slope < 0.3f) return 0.0f;
        
        float theta_deg = (float)(atan(abs_slope) * 180.0 / CV_PI);
        float min_theta = (float)(atan(0.3) * 180.0 / CV_PI);
        float max_theta = 90.0f;
        
        float score_magnitude = 0.0f;
        if (theta_deg >= max_theta) score_magnitude = 0.0f;
        else if (theta_deg <= min_theta) score_magnitude = 100.0f;
        else score_magnitude = 100.0f * (max_theta - theta_deg) / (max_theta - min_theta);
        
        return (slope > 0) ? score_magnitude : -score_magnitude;
    }

    vector<int> average_lines_projected(const vector<Vec4i>& lines, int y_min, int y_max) {
        if (lines.empty()) return {};
        vector<int> x_tops, x_bottoms;
        for (const auto& line : lines) {
            int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
            if (x2 == x1) {
                x_tops.push_back(x1); x_bottoms.push_back(x1);
                continue;
            }
            float slope = (float)(y2 - y1) / (x2 - x1);
            int val_x_top = (int)((y_min - y1) / slope + x1);
            int val_x_bottom = (int)((y_max - y1) / slope + x1);
            x_tops.push_back(val_x_top); x_bottoms.push_back(val_x_bottom);
        }
        long long sum_top = accumulate(x_tops.begin(), x_tops.end(), 0LL);
        long long sum_bottom = accumulate(x_bottoms.begin(), x_bottoms.end(), 0LL);
        return { (int)(sum_top / x_tops.size()), y_min, (int)(sum_bottom / x_bottoms.size()), y_max };
    }

    vector<Vec4i> filter_middle_33_percent(vector<Vec4i>& lines) {
        if (lines.size() < 3) return lines;
        sort(lines.begin(), lines.end(), [](const Vec4i& a, const Vec4i& b) {
            float mid_x_a = (a[0] + a[2]) / 2.0f;
            float mid_x_b = (b[0] + b[2]) / 2.0f;
            return mid_x_a < mid_x_b;
        });
        int total_count = lines.size();
        int start_idx = (int)(total_count * 0.33);
        int end_idx = (int)(total_count * (1.0 - 0.33));
        if (start_idx >= end_idx) return lines;
        return vector<Vec4i>(lines.begin() + start_idx, lines.begin() + end_idx);
    }

    vector<Vec4i> filter_by_position(const vector<Vec4i>& lines, string side, int img_w, int img_h) {
        if (lines.empty()) return {};
        vector<Vec4i> filtered;
        int threshold_x = img_w / 2;
        int threshold_y = (int)(img_h * 0.8);
        for (const auto& line : lines) {
            int mx = (line[0] + line[2]) / 2;
            int my = (line[1] + line[3]) / 2;
            bool is_noise = false;
            if (side == "left") { if (mx > threshold_x && my < threshold_y) is_noise = true; }
            else if (side == "right") { if (mx < threshold_x && my < threshold_y) is_noise = true; }
            if (!is_noise) filtered.push_back(line);
        }
        return filtered;
    }

    // ==========================================
    // 메인 콜백 함수
    // ==========================================
    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        double start_time = ros::Time::now().toSec();
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        Mat frame = cv_ptr->image;
        int h = frame.rows;
        int w = frame.cols;
        int cx = w / 2;

        // 1. Pre-processing
        cvtColor(frame, gray_img, COLOR_BGR2GRAY);
        GaussianBlur(gray_img, blur_img, Size(7, 7), 0);
        Canny(blur_img, edge_img, 100, 200);

        // 2. ROI Logic
        float roi_height = 0.45;
        int y_top = (int)(h * (1 - roi_height));
        int y_mid = (int)(h * (1 - roi_height / 2));

        Point roi_points[1][4];
        roi_points[0][0] = Point(cx - (int)(w * 0.5), h);
        roi_points[0][1] = Point(cx - (int)(w * 0.45), y_top);
        roi_points[0][2] = Point(cx + (int)(w * 0.45), y_top);
        roi_points[0][3] = Point(cx + (int)(w * 0.5), h);
        const Point* ppt[1] = { roi_points[0] };
        int npt[] = { 4 };

        if (roi_mask.empty() || roi_mask.size() != frame.size()) {
            roi_mask = Mat::zeros(h, w, CV_8UC1);
        } else {
            roi_mask.setTo(Scalar(0));
        }
        fillPoly(roi_mask, ppt, npt, 1, Scalar(255));

        int hood_h = (int)(h * 0.1);
        int hood_w_half = (int)((w * 0.50) / 2);
        Point hood_top_left(cx - hood_w_half, h - hood_h);
        Point hood_bottom_right(cx + hood_w_half, h);
        rectangle(roi_mask, hood_top_left, hood_bottom_right, Scalar(0), -1);

        Mat roi_applied;
        bitwise_and(edge_img, roi_mask, roi_applied);

        // -------------------------------------------------------------
        // [복구 완료] 팽창 (Dilate) 적용
        // Python: self.cuda_dilate.apply(gpu_edges) (iterations=3)
        // -------------------------------------------------------------
        dilate(roi_applied, roi_applied, kernel_dilate, Point(-1, -1), 3);
        // -------------------------------------------------------------

        // 3. Crossline Detection
        // 팽창된 이미지를 기준으로 픽셀 수 계산 (Python과 동일 순서)
        int white_pixel_area = countNonZero(roi_applied);
        int is_detected_now = (white_pixel_area > cross_threshold) ? 1 : 0;
        
        cross_queue.push_back(is_detected_now);
        if (cross_queue.size() > cross_queue_maxlen) cross_queue.pop_front();
        
        int sum_queue = accumulate(cross_queue.begin(), cross_queue.end(), 0);
        int final_cross_status = (sum_queue > (cross_queue.size() / 2)) ? 1 : 0;
        
        std_msgs::Int16 cross_msg;
        cross_msg.data = final_cross_status;
        pub_cross.publish(cross_msg);

        // 4. Hough Transform
        vector<Vec4i> lines;
        HoughLinesP(roi_applied, lines, 1, CV_PI / 180, 50, 50, 5);

        Mat mask_bgr;
        cvtColor(roi_applied, mask_bgr, COLOR_GRAY2BGR);

        // 5. Filter & Lane Compute
        float filtering_slope = 0.5;
        vector<Vec4i> left_lines, right_lines;

        for (const auto& line : lines) {
            int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
            int dx = x2 - x1;
            int dy = y2 - y1;
            
            if (dx == 0) continue;
            float slope = (float)dy / dx;
            if (abs(slope) <= filtering_slope) continue;

            float ls = calculate_line_score(x1, y1, x2, y2);
            float lm = calculate_midpoint_score(x1, x2, w);

            if (ls + lm <= 0) left_lines.push_back(line);
            else right_lines.push_back(line);
        }

        left_lines = filter_by_position(left_lines, "left", w, h);
        right_lines = filter_by_position(right_lines, "right", w, h);
        left_lines = filter_middle_33_percent(left_lines);
        right_lines = filter_middle_33_percent(right_lines);

        // Visualization
        for (auto& l : left_lines) line(mask_bgr, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1);
        for (auto& l : right_lines) line(mask_bgr, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 1);

        vector<int> left_result = average_lines_projected(left_lines, y_top, h);
        vector<int> right_result = average_lines_projected(right_lines, y_top, h);

        int left_mid_point = 0;
        int right_mid_point = w;

        if (!left_result.empty()) {
            int lx1 = left_result[0], ly1 = left_result[1], lx2 = left_result[2], ly2 = left_result[3];
            if (ly2 - ly1 != 0) left_mid_point = (int)((y_mid - ly1) * (double)(lx2 - lx1) / (ly2 - ly1) + lx1);
            else left_mid_point = lx1;
            line(mask_bgr, Point(lx1, ly1), Point(lx2, ly2), Scalar(0, 0, 255), 3);
        }

        if (!right_result.empty()) {
            int lx1 = right_result[0], ly1 = right_result[1], lx2 = right_result[2], ly2 = right_result[3];
            if (ly2 - ly1 != 0) right_mid_point = (int)((y_mid - ly1) * (double)(lx2 - lx1) / (ly2 - ly1) + lx1);
            else right_mid_point = lx1;
            line(mask_bgr, Point(lx1, ly1), Point(lx2, ly2), Scalar(255, 0, 0), 3);
        }

        if (!left_result.empty() && !right_result.empty()) {
            int min_lane_width = 200;
            if (left_mid_point > (right_mid_point - min_lane_width)) left_mid_point = right_mid_point - min_lane_width;
            if (right_mid_point < (left_mid_point + min_lane_width)) right_mid_point = left_mid_point + min_lane_width;
        }

        int final_midpoint = (left_mid_point + right_mid_point) / 2;
        int image_center_x = w / 2;
        
        steer_history.push_back(final_midpoint);
        if (steer_history.size() > window_size) steer_history.pop_front();

        int filtered_midpoint = final_midpoint;
        if (!steer_history.empty()) {
            long long sum = accumulate(steer_history.begin(), steer_history.end(), 0LL);
            filtered_midpoint = (int)(sum / steer_history.size());
        }

        int pubdata = (int)(-(filtered_midpoint - image_center_x) * 0.2);
        
        std_msgs::Int16 steer_msg;
        steer_msg.data = pubdata;
        pub_steer.publish(steer_msg);

        // Publish
        std_msgs::Int32MultiArray lanes_msg;
        lanes_msg.data.resize(8, -1);
        if (!left_result.empty()) copy(left_result.begin(), left_result.end(), lanes_msg.data.begin());
        if (!right_result.empty()) copy(right_result.begin(), right_result.end(), lanes_msg.data.begin() + 4);
        pub_lines_px.publish(lanes_msg);

        geometry_msgs::PointStamped target_msg;
        target_msg.header.stamp = ros::Time::now();
        target_msg.header.frame_id = "camera_frame";
        target_msg.point.x = filtered_midpoint;
        target_msg.point.y = y_mid;
        pub_target_px.publish(target_msg);

        // Visualization
        polylines(mask_bgr, ppt, npt, 1, true, Scalar(0, 255, 0), 2);
        line(mask_bgr, Point(cx, 0), Point(cx, h), Scalar(255, 255, 255), 1);

        string area_text = "White Area: " + to_string(white_pixel_area);
        int baseline = 0;
        Size textSize = getTextSize(area_text, FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
        putText(mask_bgr, area_text, Point(w - textSize.width - 20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);

        rectangle(mask_bgr, hood_top_left, hood_bottom_right, Scalar(0, 0, 255), 2);
        putText(mask_bgr, "Hood Mask", Point(hood_top_left.x, hood_top_left.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);

        circle(mask_bgr, Point(filtered_midpoint, y_mid), 20, Scalar(255, 255, 0), -1);
        putText(mask_bgr, "Offset: " + to_string(pubdata), Point(filtered_midpoint - 80, y_mid - 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

        circle(mask_bgr, Point(left_mid_point, y_mid), 20, Scalar(0, 0, 255), -1);
        circle(mask_bgr, Point(right_mid_point, y_mid), 20, Scalar(255, 0, 0), -1);

        if (final_cross_status == 1) {
            string warning_text = "CROSSLINE DETECTED";
            Size warnSize = getTextSize(warning_text, FONT_HERSHEY_SIMPLEX, 1.2, 3, &baseline);
            putText(mask_bgr, warning_text, Point(cx - warnSize.width / 2, h / 2), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 0, 255), 3);
        }

        {
            lock_guard<mutex> lock(display_mutex);
            mask_bgr.copyTo(display_img);
            new_display_data = true;
        }

        double end_time = ros::Time::now().toSec();
        int elapsed_ms = (int)((end_time - start_time) * 1000);
        if (elapsed_ms > 10) cout << "[Lag Warning] Loop Time: " << elapsed_ms << "ms" << endl;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lane_detector_cpu");
    LaneDetector ld;
    ros::spin();
    return 0;
}