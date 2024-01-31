
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{

    std::string filename = "";
    std::cout << "Which .pcd file do you want to compute the normals for?" << std::endl;
    std::cin >> filename;
    std::cout << "Processing: " << filename << std::endl;
    cv::Mat mat_depth = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat zerochannel = cv::Mat::zeros(cv::Size(mat_depth.rows, mat_depth.cols), CV_16U);
    cv::Mat output = cv::Mat::zeros(mat_depth.rows, mat_depth.cols, CV_16UC3);
    cv::Mat images[3] = {mat_depth, mat_depth, mat_depth};
    int dims[3] = {2, mat_depth.rows, mat_depth.cols};
    cv::Mat joined(3, dims, CV_16U);
    for (int i = 0; i < 3; i++)
    {
        uint16_t *ptr = &joined.at<uint16_t>(i, 0, 0);                            // pointer to first element of slice i
        cv::Mat destination(mat_depth.rows, mat_depth.cols, CV_32S, (void *)ptr); // no data copy, see documentation
        images[i].copyTo(destination);
    }

    for (int x = 0; x < images[0].rows; x++)
    {
        for (int y = 0; y < images[0].cols; y++)
        {
            output.at<cv::Vec3s>(x, y)[2] = images[0].at<unsigned short>(x, y);
            output.at<cv::Vec3s>(x, y)[1] = images[1].at<unsigned short>(x, y);
            output.at<cv::Vec3s>(x, y)[0] = images[2].at<unsigned short>(x, y);
        }
    }
    filename = filename.substr(0, filename.size() - 4);
    cv::imwrite(filename + "_d3.png", output);

    return (0);
}
