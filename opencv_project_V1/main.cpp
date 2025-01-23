#include <iostream>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/video/tracking.hpp>
#include <chrono> 
#include <map>
#include <string>
//#include "SerialPort.h"

using namespace std;
using namespace cv;
//int port;
//CSerialPort mySerialPort;
//unsigned char pData[5] = { 0 };
//#define HEAD_FRAME 0XFD
//#define END_FRAME 0XDE

// 声明全局变量
cv::KalmanFilter KF(4, 2, 0); // 创建卡尔曼滤波器，状态向量为4维，观测向量为2维
cv::Mat_<float> state(4, 1); // 状态向量，包含x, y, dx/dt, dy/dt
cv::Mat_<float> measurement(2, 1); // 观测向量，包含x, y
bool isInitialized = false; // 标志是否已经初始化卡尔曼滤波器

std::chrono::high_resolution_clock::time_point lastBlinkStart, lastYawnStart;
bool eyeClosed = false;
bool yawnDetected = false;
double eyeClosedDuration = 0, yawnDuration = 0;
int fatigue_state = 0;//默认为没检测到人脸
// 定义阈值
const double EAR_DANGER_THRESHOLD = 0.16;
const double EAR_WARNING_THRESHOLD = 0.22;
const double MAR_YAWN_THRESHOLD = 0.78;
const double MAR_SPEAK_THRESHOLD = 0.5;
const double EYE_DURATION_THRESHOLD = 1.5; // 闭眼时间阈值，单位秒
const double YAWN_DURATION_THRESHOLD = 3.0; // 打哈欠时间阈值，单位秒
const double HIGH_FATIGUE_THRESHOLD = 0.8;

//定义变量
double yawningLevel;
double speakingLevel;
double closingLevel;
double normalLevel;
double mediumLevel;
double fatigueLevel;

//void PortOpen() //打开串口
//{
//    cout << "Please insert your port number : " << endl;
//    cin >> port;
//
//    if (!mySerialPort.InitPort(port, 115200, 'N', 8, 1, EV_RXCHAR))
//    {
//        std::cout << "initPort fail !" << std::endl;
//        PortOpen();
//    }
//    else
//    {
//        std::cout << "initPort success !" << std::endl;
//    }
//}

// 计算EAR值
double eyeAspectRatio(const std::vector<dlib::point>& landmarks) {
    double horizontal = dlib::length(landmarks[1] - landmarks[5]) + dlib::length(landmarks[2] - landmarks[4]);
    double vertical = dlib::length(landmarks[0] - landmarks[3]);
    return horizontal / (2.0 * vertical);
}


double mouthAspectRatio(const std::vector<cv::Point>& mouth) {
    double A = cv::norm(mouth[2] - mouth[9]);  // 51, 59
    double B = cv::norm(mouth[4] - mouth[7]);  // 53, 57
    double C = cv::norm(mouth[0] - mouth[6]);  // 49, 55
    return (A + B) / (2.0 * C);
}

std::map<std::string, double> calculateMBBA( double mar, double yawnDuration) {
    std::map<std::string, double> mbba{
        {"Yawning", 0.0},
        {"Speaking", 0.0},
        {"Closing", 0.0}
    };

    // 根据 MAR 值分配信任度
    if (mar < MAR_SPEAK_THRESHOLD) 
    {
        mbba["Yawning"] = 0.05; //闭嘴
        mbba["Speaking"] = 0.05;
        mbba["Closing"] = 0.9;
    }
    else if(mar> MAR_SPEAK_THRESHOLD&&mar<MAR_YAWN_THRESHOLD)
    {
        mbba["Speaking"] = 0.9; // 正常交谈
        mbba["Yawning"] = 0.05;
        mbba["Closing"] = 0.05;
    }

    
    else if (mar > MAR_YAWN_THRESHOLD)
    {
        if (yawnDuration > YAWN_DURATION_THRESHOLD) {
            mbba["Yawning"] = 0.98;  // 增加打哈欠的信任度
            mbba["Speaking"] = 0.01;
            mbba["Closing"] = 0.01;
        }
        else
        {
            mbba["Yawning"] = 0.9;  // 增加打哈欠的信任度
            mbba["Speaking"] = 0.05;
            mbba["Closing"] = 0.05;
        }
    }
    
    double totalBBA = 0.0;
    for (auto& pair : mbba) {
        totalBBA += pair.second;
    }

    if (totalBBA > 1.0) {
        for (auto& pair : mbba) {
            pair.second /= totalBBA;
        }
    }
    return mbba;
}
std::map<std::string, double> calculateEBBA(double ear, double eyeClosedDuration) {
    std::map<std::string, double> ebba{
        {"NORMAL", 0.0},
        {"MEDIUM", 0.0},
        {"FATIGUE", 0.0}
    };

    //// 根据 EAR 值分配信任度
    if (ear > EAR_WARNING_THRESHOLD)
    {
        ebba["NORMAL"] = 0.9; //睁眼
        ebba["MEDIUM"] = 0.05;
        ebba["FATIGUE"] = 0.05;
    }
    else if (ear < EAR_WARNING_THRESHOLD && ear > EAR_DANGER_THRESHOLD)
    {
        ebba["NORMAL"] = 0.005; //眨眼
        ebba["MEDIUM"] = 0.99;
        ebba["FATIGUE"] = 0.005;
    }
    else if (ear < EAR_DANGER_THRESHOLD)
    {
        if (eyeClosedDuration > EYE_DURATION_THRESHOLD) {
            ebba["NORMAL"] = 0.01; //闭眼
            ebba["MEDIUM"] = 0.01;
            ebba["FATIGUE"] = 0.98;
        }
        else
        {
            ebba["NORMAL"] = 0.05; //闭眼
            ebba["MEDIUM"] = 0.05;
            ebba["FATIGUE"] = 0.9;
        }
    }
    
    double totalBBA = 0.0;
    for (auto& pair : ebba) {
        totalBBA += pair.second;
    }

    if (totalBBA > 1.0) {
        for (auto& pair : ebba) {
            pair.second /= totalBBA;
        }
    }
    return ebba;
}

std::map<std::string, double> combineBBA(const std::map<std::string, double>& bba1, const std::map<std::string, double>& bba2) {
    std::map<std::string, double> combinedBBA;
    double totalWeight = 0.0;

    // 直接合并相同标签的信任度
    for (const auto& e1 : bba1) {
        for (const auto& e2 : bba2) {
            if (e1.first == e2.first) {
                combinedBBA[e1.first] += e1.second * e2.second;
                totalWeight += e1.second * e2.second;
            }
        }
    }

    // 归一化处理
    for (auto& bba : combinedBBA) {
        bba.second /= totalWeight;
    }

    return combinedBBA;
}

void makeMDecision(const std::map<std::string, double>& combinedBBA) {
    // 获取信任度
     yawningLevel = combinedBBA.at("Yawning");
     speakingLevel = combinedBBA.at("Speaking");
    closingLevel = combinedBBA.at("Closing");

    
    /*cout << "Yawning: " << yawningLevel << endl;
    cout << "Speaking: " << speakingLevel << endl;
    cout << "Closing: " << closingLevel << endl;*/

    
}
void makeEDecision(const std::map<std::string, double>& combinedBBA)
{
    // 获取信任度
    normalLevel = combinedBBA.at("NORMAL");
    mediumLevel = combinedBBA.at("MEDIUM");
    fatigueLevel = combinedBBA.at("FATIGUE");


    /*cout << "NORMAL: " << normalLevel << endl;
    cout << "MEDIUM: " << mediumLevel << endl;
    cout << "FATIGUE: " << fatigueLevel << endl;*/
}
void maketotalDecision()
{
   
    if ((yawningLevel > HIGH_FATIGUE_THRESHOLD && fatigueLevel > HIGH_FATIGUE_THRESHOLD)|| (yawningLevel > HIGH_FATIGUE_THRESHOLD && fatigueLevel < HIGH_FATIGUE_THRESHOLD) || (yawningLevel <  HIGH_FATIGUE_THRESHOLD && fatigueLevel > HIGH_FATIGUE_THRESHOLD))
    {
        fatigue_state = 1;//疲劳
    }
    else if (mediumLevel > HIGH_FATIGUE_THRESHOLD)
    {
        fatigue_state = 2;//轻度 
    }
    else if (speakingLevel > HIGH_FATIGUE_THRESHOLD)
    {
        fatigue_state = 3;//说话
    }
    else
    {
        fatigue_state= 4;//不疲劳
    }
  /*  pData[3] = fatigue_state;
    mySerialPort.WriteData((unsigned char*)pData, 5);*/
   /* cout << "fatigue_state:" << fatigue_state << endl;*/
}



int main() {
    // 加载人脸检测器
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    dlib::deserialize("D:/shape_predictor_68_face_landmarks.dat") >> predictor;

    // 计时器，在每一帧中都计算经过的时间
   /* PortOpen();
    pData[0] = HEAD_FRAME;
    pData[4] = END_FRAME;
    if (!mySerialPort.OpenListenThread()) {
        std::cout << "OpenListenThread fail !" << std::endl;
    }
    else {
        std::cout << "OpenListenThread success !" << std::endl;
    }*/

    clock_t start_time = clock();   // 记录开始时间

    // 打开摄像头
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open camera." << endl;
        return -1;
    }
    Mat frame;
    bool faceDetected = false;

    // 用于跟踪是否成功检测到人脸
    int framesSinceDetection = 0;
    int framesSinceCalculation = 0;
    int framesSinceLastReset = 0;
    const int RESET_INTERVAL = 300;  // 每300帧重置一次
    double lastEar = 0.0;
    double lastMar = 0.0;
    bool danger = false;
    bool tiny_danger = false;
    int danger_mode = 0;
    int mouth_mode = 0;
    bool tiny_danger_time_set = false;
    bool danger_time_set = false;
    auto dangerTime = std::chrono::high_resolution_clock::now();
    auto tinyDangerTime = std::chrono::high_resolution_clock::now();
    auto yawnTime = std::chrono::high_resolution_clock::now();
    bool yawnInProgress = false;

    int interaction = 1;
    int danger_judge = 0;

    std::map<std::string, double> prevEBBA;
    std::map<std::string, double> prevMBBA;

    while (true) {
        // 读取当前帧
        cap.read(frame);
        framesSinceLastReset++;
        if (framesSinceLastReset >= RESET_INTERVAL)
        {
            prevMBBA.clear();
            prevEBBA.clear();
            isInitialized = false;  // 重置卡尔曼滤波器初始化标志
            framesSinceLastReset = 0;  // 重置帧数计数器
        }
        // 直方图均衡化
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(frame, frame);

        // 将OpenCV的Mat对象转换为Dlib的图像类型
        dlib::cv_image<unsigned char> dlib_img(frame);

        // 检测人脸
        std::vector<dlib::rectangle> faces = detector(dlib_img);
        dlib::full_object_detection shape;

        // 更新帧计数器
        if (faces.size() == 1) {
            faceDetected = true;
            framesSinceDetection = 0;

            // 提取关键点坐标
            dlib::full_object_detection shape = predictor(dlib_img, faces[0]);
            cv::Point2f pt(shape.part(0).x(), shape.part(0).y()); // 假设关键点0为眼睛左上角
            measurement(0) = pt.x;
            measurement(1) = pt.y;

           

            if (!isInitialized) {
                // 初始化卡尔曼滤波器
                KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);//初始化卡尔曼滤波器其他参数
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
        }
        else {
            faceDetected = false;
            mouth_mode = 0;  // 重置为默认值
            framesSinceDetection++;
        }
        // 在摄像头界面上显示是否成功检测到人脸以及人脸关键点
        if (faceDetected) 
        {
            
            if (framesSinceCalculation % 30 == 0) { // 每30帧计算一次EAR值
                interaction++;
                double ear = 0.0;
                double mar = 0.0;
                for (size_t i = 0; i < faces.size(); ++i) {
                    dlib::full_object_detection shape = predictor(dlib_img, faces[i]);

                    for (size_t j = 0; j < shape.num_parts(); ++j) {
                        cv::Point point(shape.part(j).x(), shape.part(j).y());
                        cv::circle(frame, point, 2, cv::Scalar(0, 255, 0), -1); // 在图像上绘制特征点
                    }
                }
                for (size_t i = 0; i < faces.size(); ++i) {
                    dlib::full_object_detection shape = predictor(dlib_img, faces[i]);
                    double ear_left = eyeAspectRatio({ shape.part(36), shape.part(37), shape.part(38), shape.part(39), shape.part(40), shape.part(41) });
                    double ear_right = eyeAspectRatio({ shape.part(42), shape.part(43), shape.part(44), shape.part(45), shape.part(46), shape.part(47) });
                    ear = (ear_left + ear_right) / 2.0;
                    std::vector<cv::Point> mouth_points;
                    for (size_t j = 48; j <= 59; ++j) {
                        cv::Point point(shape.part(j).x(), shape.part(j).y());
                        mouth_points.push_back(point);
                    }
                    double mar = mouthAspectRatio(mouth_points);
                    lastMar = mar; // 将计算得到的 MAR 值赋给 lastMar
                }
                lastEar = ear; // 更新最后的EAR值

                // 检查闭眼持续时间
                if (ear < EAR_DANGER_THRESHOLD && !eyeClosed) {
                    lastBlinkStart = std::chrono::high_resolution_clock::now();
                    eyeClosed = true;
                }
                else if (ear >= EAR_DANGER_THRESHOLD && eyeClosed) {
                    auto now = std::chrono::high_resolution_clock::now();
                    eyeClosedDuration = std::chrono::duration_cast<std::chrono::seconds>(now - lastBlinkStart).count();
                    eyeClosed = false;
                    /*cout << "Eye closed duration: " << eyeClosedDuration << " seconds" << endl;*/
                }

                // 检查打哈欠持续时间
                if (mar > MAR_YAWN_THRESHOLD && !yawnDetected) {
                    lastYawnStart = std::chrono::high_resolution_clock::now();
                    yawnDetected = true;
                }
                else if (mar <= MAR_YAWN_THRESHOLD && yawnDetected) {
                    auto now = std::chrono::high_resolution_clock::now();
                    yawnDuration = std::chrono::duration_cast<std::chrono::seconds>(now - lastYawnStart).count();
                    yawnDetected = false;
                    /*cout << "Yawn duration: " << yawnDuration << " seconds" << endl;*/
                }

                std::map<std::string, double> currentMBBA = calculateMBBA( lastMar, yawnDuration);
                std::map<std::string, double> currentEBBA = calculateEBBA( lastEar, eyeClosedDuration);
                if (prevMBBA.empty()) {
                    prevMBBA = currentMBBA;
                }
                else {
                    // 合并BBA
                    std::map<std::string, double> combinedMBBA = combineBBA(prevMBBA, currentMBBA);
                    prevMBBA = combinedMBBA;  // 更新prevBBA为合并后的结果
                }
                
                if (prevEBBA.empty()) {
                    prevEBBA = currentEBBA;
                }
                else {
                    // 合并BBA
                    std::map<std::string, double> combinedEBBA = combineBBA(prevEBBA, currentEBBA);
                    prevEBBA = combinedEBBA;  // 更新prevBBA为合并后的结果
                }
                makeMDecision(prevMBBA);
                makeEDecision(prevEBBA);
                maketotalDecision();
                // 显示计算结果和状态信息
                putText(frame, "EAR: " + to_string(lastEar), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
                putText(frame, "MAR: " + to_string(lastMar), Point(10, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
                
            }
            framesSinceCalculation = 0;
        }
        else 
        {
            putText(frame, "No face detected or too much face.", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
            putText(frame, "", Point(10, 90), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
            fatigue_state = 0;
          /*  pData[3] = fatigue_state;
            mySerialPort.WriteData((unsigned char*)pData, 5);*/
           /* cout << "fatigue_state:" << fatigue_state << endl;*/
            framesSinceCalculation++;
        }
        putText(frame, "STATE: " + to_string(fatigue_state), Point(10, 90), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        if (fatigue_state == 0)
        {
            putText(frame, "No face detected.", Point(10, 120), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }
        else if (fatigue_state == 1)
        {
            putText(frame, " fatigue.", Point(10, 120), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }
        else if (fatigue_state == 2)
        {
            putText(frame, "Mild fatigue.", Point(10, 120), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }
        else if (fatigue_state == 3)
        {
            putText(frame, "Speaking..", Point(10, 120), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }
        else if (fatigue_state == 4)
        {
            putText(frame, "Not fatigue..", Point(10, 120), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }
        imshow("Face Detection", frame);
        // 按下ESC键退出循环
        if (waitKey(1) == 27) { break; }
    } // End of while loop

    // 释放摄像头资源
    cap.release();
    // 关闭所有显示窗口
    destroyAllWindows();
    return 0;
}