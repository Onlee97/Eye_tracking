#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    VideoCapture cap(0); // open the video camera no. 0

    if (!cap.isOpened())  // if not success, exit program
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }


    namedWindow("MyVideo",WINDOW_AUTOSIZE);

    while (1)
    {
        Mat frame;

        bool bSuccess = cap.read(frame); // read a new frame from video

         if (!bSuccess)
        {
             cout << "Cannot read a frame from video stream" << endl;
             break;
        }

        Mat grayscale;
        cvtColor(frame, grayscale, COLOR_RGB2GRAY);

        imshow("MyVideo", grayscale);

        if (waitKey(30) == 27)
       {
            cout << "esc key is pressed by user" << endl;
            break;
       }
    }
    return 0;

}
