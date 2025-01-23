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

// ����ȫ�ֱ���
cv::KalmanFilter KF(4, 2, 0); // �����������˲�����״̬����Ϊ4ά���۲�����Ϊ2ά
cv::Mat_<float> state(4, 1); // ״̬����������x, y, dx/dt, dy/dt
cv::Mat_<float> measurement(2, 1); // �۲�����������x, y
bool isInitialized = false; // ��־�Ƿ��Ѿ���ʼ���������˲���

std::chrono::high_resolution_clock::time_point lastBlinkStart, lastYawnStart;
bool eyeClosed = false;
bool yawnDetected = false;
double eyeClosedDuration = 0, yawnDuration = 0;
int fatigue_state = 0;//Ĭ��Ϊû��⵽����
// ������ֵ
const double EAR_DANGER_THRESHOLD = 0.16;
const double EAR_WARNING_THRESHOLD = 0.22;
const double MAR_YAWN_THRESHOLD = 0.78;
const double MAR_SPEAK_THRESHOLD = 0.5;
const double EYE_DURATION_THRESHOLD = 1.5; // ����ʱ����ֵ����λ��
const double YAWN_DURATION_THRESHOLD = 3.0; // ���Ƿʱ����ֵ����λ��
const double HIGH_FATIGUE_THRESHOLD = 0.8;

//�������
double yawningLevel;
double speakingLevel;
double closingLevel;
double normalLevel;
double mediumLevel;
double fatigueLevel;

//void PortOpen() //�򿪴���
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

// ����EARֵ
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

    // ���� MAR ֵ�������ζ�
    if (mar < MAR_SPEAK_THRESHOLD) 
    {
        mbba["Yawning"] = 0.05; //����
        mbba["Speaking"] = 0.05;
        mbba["Closing"] = 0.9;
    }
    else if(mar> MAR_SPEAK_THRESHOLD&&mar<MAR_YAWN_THRESHOLD)
    {
        mbba["Speaking"] = 0.9; // ������̸
        mbba["Yawning"] = 0.05;
        mbba["Closing"] = 0.05;
    }

    
    else if (mar > MAR_YAWN_THRESHOLD)
    {
        if (yawnDuration > YAWN_DURATION_THRESHOLD) {
            mbba["Yawning"] = 0.98;  // ���Ӵ��Ƿ�����ζ�
            mbba["Speaking"] = 0.01;
            mbba["Closing"] = 0.01;
        }
        else
        {
            mbba["Yawning"] = 0.9;  // ���Ӵ��Ƿ�����ζ�
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

    //// ���� EAR ֵ�������ζ�
    if (ear > EAR_WARNING_THRESHOLD)
    {
        ebba["NORMAL"] = 0.9; //����
        ebba["MEDIUM"] = 0.05;
        ebba["FATIGUE"] = 0.05;
    }
    else if (ear < EAR_WARNING_THRESHOLD && ear > EAR_DANGER_THRESHOLD)
    {
        ebba["NORMAL"] = 0.005; //գ��
        ebba["MEDIUM"] = 0.99;
        ebba["FATIGUE"] = 0.005;
    }
    else if (ear < EAR_DANGER_THRESHOLD)
    {
        if (eyeClosedDuration > EYE_DURATION_THRESHOLD) {
            ebba["NORMAL"] = 0.01; //����
            ebba["MEDIUM"] = 0.01;
            ebba["FATIGUE"] = 0.98;
        }
        else
        {
            ebba["NORMAL"] = 0.05; //����
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

    // ֱ�Ӻϲ���ͬ��ǩ�����ζ�
    for (const auto& e1 : bba1) {
        for (const auto& e2 : bba2) {
            if (e1.first == e2.first) {
                combinedBBA[e1.first] += e1.second * e2.second;
                totalWeight += e1.second * e2.second;
            }
        }
    }

    // ��һ������
    for (auto& bba : combinedBBA) {
        bba.second /= totalWeight;
    }

    return combinedBBA;
}

void makeMDecision(const std::map<std::string, double>& combinedBBA) {
    // ��ȡ���ζ�
     yawningLevel = combinedBBA.at("Yawning");
     speakingLevel = combinedBBA.at("Speaking");
    closingLevel = combinedBBA.at("Closing");

    
    /*cout << "Yawning: " << yawningLevel << endl;
    cout << "Speaking: " << speakingLevel << endl;
    cout << "Closing: " << closingLevel << endl;*/

    
}
void makeEDecision(const std::map<std::string, double>& combinedBBA)
{
    // ��ȡ���ζ�
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
        fatigue_state = 1;//ƣ��
    }
    else if (mediumLevel > HIGH_FATIGUE_THRESHOLD)
    {
        fatigue_state = 2;//��� 
    }
    else if (speakingLevel > HIGH_FATIGUE_THRESHOLD)
    {
        fatigue_state = 3;//˵��
    }
    else
    {
        fatigue_state= 4;//��ƣ��
    }
  /*  pData[3] = fatigue_state;
    mySerialPort.WriteData((unsigned char*)pData, 5);*/
   /* cout << "fatigue_state:" << fatigue_state << endl;*/
}



int main() {
    // �������������
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    dlib::deserialize("D:/shape_predictor_68_face_landmarks.dat") >> predictor;

    // ��ʱ������ÿһ֡�ж����㾭����ʱ��
   /* PortOpen();
    pData[0] = HEAD_FRAME;
    pData[4] = END_FRAME;
    if (!mySerialPort.OpenListenThread()) {
        std::cout << "OpenListenThread fail !" << std::endl;
    }
    else {
        std::cout << "OpenListenThread success !" << std::endl;
    }*/

    clock_t start_time = clock();   // ��¼��ʼʱ��

    // ������ͷ
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open camera." << endl;
        return -1;
    }
    Mat frame;
    bool faceDetected = false;

    // ���ڸ����Ƿ�ɹ���⵽����
    int framesSinceDetection = 0;
    int framesSinceCalculation = 0;
    int framesSinceLastReset = 0;
    const int RESET_INTERVAL = 300;  // ÿ300֡����һ��
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
        // ��ȡ��ǰ֡
        cap.read(frame);
        framesSinceLastReset++;
        if (framesSinceLastReset >= RESET_INTERVAL)
        {
            prevMBBA.clear();
            prevEBBA.clear();
            isInitialized = false;  // ���ÿ������˲�����ʼ����־
            framesSinceLastReset = 0;  // ����֡��������
        }
        // ֱ��ͼ���⻯
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(frame, frame);

        // ��OpenCV��Mat����ת��ΪDlib��ͼ������
        dlib::cv_image<unsigned char> dlib_img(frame);

        // �������
        std::vector<dlib::rectangle> faces = detector(dlib_img);
        dlib::full_object_detection shape;

        // ����֡������
        if (faces.size() == 1) {
            faceDetected = true;
            framesSinceDetection = 0;

            // ��ȡ�ؼ�������
            dlib::full_object_detection shape = predictor(dlib_img, faces[0]);
            cv::Point2f pt(shape.part(0).x(), shape.part(0).y()); // ����ؼ���0Ϊ�۾����Ͻ�
            measurement(0) = pt.x;
            measurement(1) = pt.y;

           

            if (!isInitialized) {
                // ��ʼ���������˲���
                KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);//��ʼ���������˲�����������
                setIdentity(KF.measurementMatrix);
                setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
                setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
                setIdentity(KF.errorCovPost, Scalar::all(1));
                // ��ʼ��״̬�����Ͳ�������
                KF.statePost.at<float>(0) = pt.x;
                KF.statePost.at<float>(1) = pt.y;
                KF.statePost.at<float>(2) = 0;
                KF.statePost.at<float>(3) = 0;
                isInitialized = true;
            }
            // Ԥ��״̬
            Mat prediction = KF.predict();
            // ���²���
            Mat estimated = KF.correct(measurement);
            // ����Ԥ��λ��
            pt.x = estimated.at<float>(0);
            pt.y = estimated.at<float>(1);
        }
        else {
            faceDetected = false;
            mouth_mode = 0;  // ����ΪĬ��ֵ
            framesSinceDetection++;
        }
        // ������ͷ��������ʾ�Ƿ�ɹ���⵽�����Լ������ؼ���
        if (faceDetected) 
        {
            
            if (framesSinceCalculation % 30 == 0) { // ÿ30֡����һ��EARֵ
                interaction++;
                double ear = 0.0;
                double mar = 0.0;
                for (size_t i = 0; i < faces.size(); ++i) {
                    dlib::full_object_detection shape = predictor(dlib_img, faces[i]);

                    for (size_t j = 0; j < shape.num_parts(); ++j) {
                        cv::Point point(shape.part(j).x(), shape.part(j).y());
                        cv::circle(frame, point, 2, cv::Scalar(0, 255, 0), -1); // ��ͼ���ϻ���������
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
                    lastMar = mar; // ������õ��� MAR ֵ���� lastMar
                }
                lastEar = ear; // ��������EARֵ

                // �����۳���ʱ��
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

                // �����Ƿ����ʱ��
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
                    // �ϲ�BBA
                    std::map<std::string, double> combinedMBBA = combineBBA(prevMBBA, currentMBBA);
                    prevMBBA = combinedMBBA;  // ����prevBBAΪ�ϲ���Ľ��
                }
                
                if (prevEBBA.empty()) {
                    prevEBBA = currentEBBA;
                }
                else {
                    // �ϲ�BBA
                    std::map<std::string, double> combinedEBBA = combineBBA(prevEBBA, currentEBBA);
                    prevEBBA = combinedEBBA;  // ����prevBBAΪ�ϲ���Ľ��
                }
                makeMDecision(prevMBBA);
                makeEDecision(prevEBBA);
                maketotalDecision();
                // ��ʾ��������״̬��Ϣ
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
        // ����ESC���˳�ѭ��
        if (waitKey(1) == 27) { break; }
    } // End of while loop

    // �ͷ�����ͷ��Դ
    cap.release();
    // �ر�������ʾ����
    destroyAllWindows();
    return 0;
}