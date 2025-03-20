#include <jni.h>
#include <android/log.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <iomanip>
#include <sstream>
#include <vector>
#include <iostream>

extern "C" void JNICALL
Java_com_example_opencv2_MainActivity_detectMarkers(
        JNIEnv *env,
        jobject it,
        jlong inAddr // in_out
) {
    const float markerLength = 0.05f;
    cv::Mat &inframe = *(cv::Mat *)(inAddr);

    cv::cvtColor(inframe, inframe, cv::COLOR_RGB2BGR);

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary (cv::aruco::DICT_7X7_50);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);
    detector.detectMarkers(inframe, markerCorners, markerIds, rejectedCandidates);

    cv::Mat camMatrix = (cv::Mat_<double>(3,3) << 628.158, 0., 324.099, 0., 628.156, 260.908, 0., 0., 1.);
    cv::Mat distCoeffs = (cv::Mat_<double>(5,1) << 0.0995485, -0.206384, 0.00754589, 0.00336531, 0);

    cv::Mat objPoints(4, 1, CV_32FC3);
    objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength/2.f, markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength/2.f, markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength/2.f, -markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength/2.f, -markerLength/2.f, 0);

    size_t nMarkers = markerCorners.size();
    std::vector<cv::Vec3d> rvecs(nMarkers), tvecs(nMarkers);

    if(!markerIds.empty()) {
        // Calculate pose for each marker
        for (size_t i = 0; i < nMarkers; i++) {
            solvePnP(objPoints, markerCorners.at(i), camMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
        }
    }

    if(!markerIds.empty()) {
        cv::aruco::drawDetectedMarkers(inframe, markerCorners, markerIds);

        for(size_t i = 0; i < nMarkers; i++) {
            cv::drawFrameAxes(inframe, camMatrix, distCoeffs, rvecs[i], tvecs[i],
                              markerLength * 1.5f, 2);
        }
    }

    cv::cvtColor(inframe, inframe, cv::COLOR_BGR2RGB);
}

