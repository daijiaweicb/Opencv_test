#include <iostream>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/video/tracking.hpp>

using namespace std;
using namespace cv;

// ����ȫ�ֱ���
cv::KalmanFilter KF( 4, 2, 0); // �����������˲�����״̬����Ϊ4ά���۲�����Ϊ2ά
cv::Mat_<float> state(4, 1); // ״̬����������x, y, dx/dt, dy/dt
cv::Mat_<float> measurement(2, 1); // �۲�����������x, y
bool isInitialized = false; // ��־�Ƿ��Ѿ���ʼ���������˲���

// ����EARֵ
double eyeAspectRatio(const std::vector<dlib::point>& landmarks) {
    // �����۾���ˮƽ�ʹ�ֱ����
    double horizontal = dlib::length(landmarks[1] - landmarks[5]) + dlib::length(landmarks[2] - landmarks[4]);
    double vertical = dlib::length(landmarks[0] - landmarks[3]);

    // ����EARֵ
    return horizontal / (2.0 * vertical);
}

//����MARֵ
double mouthAspectRatio(const std::vector<cv::Point>& mouth) {
    // ������߶γ���
    double A = cv::norm(mouth[2] - mouth[9]);  // 51, 59
    double B = cv::norm(mouth[4] - mouth[7]);  // 53, 57
    double C = cv::norm(mouth[0] - mouth[6]);  // 49, 55

    // ����MARֵ
    double mar = (A + B) / (2.0 * C);

    return mar;
}

int main() {
    // �������������
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    dlib::deserialize("D:/shape_predictor_68_face_landmarks.dat") >> predictor;

    // ������ͷ
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open camera." << endl;
        return -1;
    }

    Mat frame;

    while (true) {
        // ��ȡ��ǰ֡
        cap.read(frame);

        // ֱ��ͼ���⻯
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(frame, frame);

        // �� OpenCV �� Mat ����ת��Ϊ Dlib ��ͼ������
        dlib::cv_image<unsigned char> dlib_img(frame);

        // �������
        std::vector<dlib::rectangle> faces = detector(dlib_img);

        // ������ͷ��������ʾ�Ƿ�ɹ���⵽�����Լ������ؼ���
        if (!faces.empty()) {
            dlib::full_object_detection shape = predictor(dlib_img, faces[0]); // ����ֻ��һ������

            // ��ȡ�ؼ�������
            cv::Point2f pt(shape.part(0).x(), shape.part(0).y()); // ����ؼ���0Ϊ�۾����Ͻ�
            measurement(0) = pt.x;
            measurement(1) = pt.y;

            if (!isInitialized) {
                // ��ʼ���������˲���״̬����
                KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);

                // ��ʼ���������˲�����������
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

            // ����������
            for (size_t i = 0; i < shape.num_parts(); ++i) {
                cv::Point point(shape.part(i).x(), shape.part(i).y());
                cv::circle(frame, point, 2, cv::Scalar(0, 255, 0), -1); // ��ͼ���ϻ���������
            }

        }

        // ��ʾ��ǰ֡
        imshow("Face Detection", frame);

        // ����ESC���˳�ѭ��
        if (waitKey(1) == 27) {
            break;
        }
    }

    // �ر�����ͷ
    cap.release();
    // �ر����д���
    destroyAllWindows();

    return 0;
}
