
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <opencv2/core/ocl.hpp>

#include <dlib/opencv.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace std;
using namespace dlib;
using namespace cv;

void detectFaceDlibHog(frontal_face_detector hogFaceDetector, Mat &frameDlibHog, int inHeight = 300, int inWidth = 0) {

    int frameHeight = frameDlibHog.rows;
    int frameWidth = frameDlibHog.cols;
    if (!inWidth)
        inWidth = (int) ((frameWidth / (float) frameHeight) * inHeight);

    float scaleHeight = frameHeight / (float) inHeight;
    float scaleWidth = frameWidth / (float) inWidth;

    Mat frameDlibHogSmall;
    resize(frameDlibHog, frameDlibHogSmall, Size(inWidth, inHeight));

    // Convert OpenCV image format to Dlib's image format
    cv_image<bgr_pixel> dlibIm(frameDlibHogSmall);

    // Detect faces in the image
    std::vector<dlib::rectangle> faceRects = hogFaceDetector(dlibIm);

    for (size_t i = 0; i < faceRects.size(); i++) {
        int x1 = (int) (faceRects[i].left() * scaleWidth);
        int y1 = (int) (faceRects[i].top() * scaleHeight);
        int x2 = (int) (faceRects[i].right() * scaleWidth);
        int y2 = (int) (faceRects[i].bottom() * scaleHeight);
        cv::rectangle(frameDlibHog, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), (int) (frameHeight / 150.0), 4);
    }
}

void VideoCap(const frontal_face_detector& hogFaceDetector, void (*detectFaceCb)(frontal_face_detector, Mat &, int, int)) {
    namedWindow("videoSource");

    // Read video
    VideoCapture cap(0);
    // Exit if video is not opened
    if (!cap.isOpened()) {
        cout << "Could not read video file" << endl;
        return;
    }

    Mat frame;
    while (cap.read(frame)) {
        if (frame.empty())
            break;
        // Start timer
        auto timer = (double) getTickCount();

        (*detectFaceCb)(hogFaceDetector, frame, 200, 0);

        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double) getTickCount() - timer);

        putText(frame, format("DLIB HoG ; FPS = %.2f", fps), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4,
                Scalar(0, 0, 255), 4);

        imshow("videoSource", frame);

        // Exit if ESC pressed.
        int k = waitKey(1);
        if (k == 27) {
            break;
        }
    }
}


int main(int argc, char **argv) {

    frontal_face_detector hogFaceDetector = get_frontal_face_detector();
    void (*detectFaceCb)(frontal_face_detector, Mat &, int, int) = &detectFaceDlibHog;

    VideoCap(hogFaceDetector, detectFaceCb);

    return 0;
}

