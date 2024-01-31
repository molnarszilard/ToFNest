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
    std::cout << "Which depth image do you want to add noise?" << std::endl;
    std::cin >> filename;
    std::cout << "Processing: " << filename << std::endl;
    cv::Mat mat_depth = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat noise = cv::Mat::zeros(mat_depth.rows, mat_depth.cols, CV_16UC1);
    cv::Mat mean = cv::Mat::zeros(1, 1, CV_16UC1);
    cv::Mat sigma = cv::Mat::ones(1, 1, CV_16UC1);
    cv::Mat negativ_bias = cv::Mat::ones(mat_depth.rows, mat_depth.cols, CV_16UC1);
    int s = 200;
    cv::randn(noise, mean + s, sigma * s);
    printf("%u+", mat_depth.at<uint16_t>(100, 100));
    printf("%u=", noise.at<uint16_t>(100, 100));
    cv::Mat output = cv::Mat::zeros(mat_depth.rows, mat_depth.cols, CV_16UC1);
    output = mat_depth + noise.mul(mat_depth.mul(1 / mat_depth)) - negativ_bias.mul(mat_depth.mul(1 / mat_depth)) * s;
    // cv::add(mat_depth,noise,output,mat_depth);
    printf("%u\n", output.at<uint16_t>(100, 100));
    filename = filename.substr(0, filename.size() - 4);
    cv::imwrite(filename + "_noise.png", output);

    return (0);
}
