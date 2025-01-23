#include <iostream>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/video/tracking.hpp>

using namespace std;
using namespace cv;

// 声明全局变量
cv::KalmanFilter KF( 4, 2, 0); // 创建卡尔曼滤波器，状态向量为4维，观测向量为2维
cv::Mat_<float> state(4, 1); // 状态向量，包含x, y, dx/dt, dy/dt
cv::Mat_<float> measurement(2, 1); // 观测向量，包含x, y
bool isInitialized = false; // 标志是否已经初始化卡尔曼滤波器

// 计算EAR值
double eyeAspectRatio(const std::vector<dlib::point>& landmarks) {
    // 计算眼睛的水平和垂直长度
    double horizontal = dlib::length(landmarks[1] - landmarks[5]) + dlib::length(landmarks[2] - landmarks[4]);
    double vertical = dlib::length(landmarks[0] - landmarks[3]);

    // 计算EAR值
    return horizontal / (2.0 * vertical);
}

//计算MAR值
double mouthAspectRatio(const std::vector<cv::Point>& mouth) {
    // 计算各线段长度
    double A = cv::norm(mouth[2] - mouth[9]);  // 51, 59
    double B = cv::norm(mouth[4] - mouth[7]);  // 53, 57
    double C = cv::norm(mouth[0] - mouth[6]);  // 49, 55

    // 计算MAR值
    double mar = (A + B) / (2.0 * C);

    return mar;
}

int main() {
    // 加载人脸检测器
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    dlib::deserialize("D:/shape_predictor_68_face_landmarks.dat") >> predictor;

    // 打开摄像头
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open camera." << endl;
        return -1;
    }

    Mat frame;

    while (true) {
        // 读取当前帧
        cap.read(frame);

        // 直方图均衡化
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(frame, frame);

        // 将 OpenCV 的 Mat 对象转换为 Dlib 的图像类型
        dlib::cv_image<unsigned char> dlib_img(frame);

        // 检测人脸
        std::vector<dlib::rectangle> faces = detector(dlib_img);

        // 在摄像头界面上显示是否成功检测到人脸以及人脸关键点
        if (!faces.empty()) {
            dlib::full_object_detection shape = predictor(dlib_img, faces[0]); // 假设只有一个人脸

            // 提取关键点坐标
            cv::Point2f pt(shape.part(0).x(), shape.part(0).y()); // 假设关键点0为眼睛左上角
            measurement(0) = pt.x;
            measurement(1) = pt.y;

            if (!isInitialized) {
                // 初始化卡尔曼滤波器状态矩阵
                KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);

                // 初始化卡尔曼滤波器其他参数
                setIdentity(KF.measurementMatrix);
                setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
                setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
                setIdentity(KF.errorCovPost, Scalar::all(1));

                // 初始化状态向量和测量向量
                KF.statePost.at<float>(0) = pt.x;
                KF.statePost.at<float>(1) = pt.y;
                KF.statePost.at<float>(2) = 0;
                KF.statePost.at<float>(3) = 0;

                isInitialized = true;
            }

            // 预测状态
            Mat prediction = KF.predict();

            // 更新测量
            Mat estimated = KF.correct(measurement);

            // 更新预测位置
            pt.x = estimated.at<float>(0);
            pt.y = estimated.at<float>(1);

            // 绘制特征点
            for (size_t i = 0; i < shape.num_parts(); ++i) {
                cv::Point point(shape.part(i).x(), shape.part(i).y());
                cv::circle(frame, point, 2, cv::Scalar(0, 255, 0), -1); // 在图像上绘制特征点
            }

        }

        // 显示当前帧
        imshow("Face Detection", frame);

        // 按下ESC键退出循环
        if (waitKey(1) == 27) {
            break;
        }
    }

    // 关闭摄像头
    cap.release();
    // 关闭所有窗口
    destroyAllWindows();

    return 0;
}
