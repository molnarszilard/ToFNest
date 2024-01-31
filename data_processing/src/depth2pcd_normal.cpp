#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <string>

using namespace std;

string directory = "";

void depth2pcd_normal(char line[])
{
    printf("Processing image %s\n", line);
    char file_depth[200];
    char file_normal[200];
    
    sprintf(file_depth, "%sdepth/%s.png", directory.c_str(), line);        
    sprintf(file_normal, "%spredictions/%s_pred.png", directory.c_str(), line);
    std::cout<<file_depth<<std::endl;
    std::cout<<file_normal<<std::endl;
    // MODIFY THE K matrix according to your camera, here is an example with a pico sense camera
    double K[9] = {460.58518931365654, 0.0, 334.0805877590529, 0.0, 460.2679961517268, 169.80766383231037, 0.0, 0.0, 1.0};
    double fx = K[0];
    double fy = K[4];
    double x0 = K[2];
    double y0 = K[5];

    // Point Clouds to hold output
    cv::Mat depth = cv::imread(file_depth, cv::IMREAD_UNCHANGED);
    cv::Mat normal = cv::imread(file_normal, cv::IMREAD_COLOR);
    std::cout<<directory<<std::endl;
    std::cout<<depth.rows<<std::endl;
    std::cout<<normal.rows<<std::endl;
    cv::Mat &mat_normal = normal;

    pcl::PointCloud<pcl::PointNormal> cloud_msg;
    pcl::PointNormal p;
    bool nan = false;
    for (int i = 0; i < depth.rows; i++)
    {
        for (int j = 0; j < depth.cols; j++)
        {
            int index = i * depth.cols + j;
            float d = depth.at<uint16_t>(i, j) / 1000.0;

            if (d == 0.0)
            {
                continue;
            }
            float x_over_z = (j - x0) / fx;
            float y_over_z = (i - y0) / fy;
            p.z = d;
            p.x = x_over_z * p.z;
            p.y = y_over_z * p.z;

            if (p.x != p.x || p.y != p.y || p.z != p.z || std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z))
                nan = true;
            if (!nan)
            {
                p.normal_x = (double)(mat_normal.ptr(i, j)[2] * 2.0 / 255.0 - 1.0);
                p.normal_y = (double)(mat_normal.ptr(i, j)[1] * 2.0 / 255.0 - 1.0);
                p.normal_z = (double)(mat_normal.ptr(i, j)[0] * 2.0 / 255.0 - 1.0);
                cloud_msg.points.push_back(p);
            }
            nan = false;
        }
    }
    cloud_msg.width = cloud_msg.points.size();
    cout << "Number of points:" << cloud_msg.width << endl;
    cloud_msg.height = 1;
    cloud_msg.points.resize(cloud_msg.width * cloud_msg.height);
    cloud_msg.is_dense = false;
    char file_pcd[200];
    sprintf(file_pcd, "%spcdpred/%s.pcd", directory.c_str(), line);

    pcl::io::savePCDFile(file_pcd, cloud_msg, true);
    cout << "[*] Conversion finished!" << endl;
}

int main(int argc, char **argv)
{
    std::cout << "Directory where you depth/ and predictions/ folder?" << std::endl;
    std::cin >> directory;
    ifstream myfile(directory+"filelist.txt");
    if (myfile.is_open())
    {
        string line;
        while (getline(myfile, line))
        {
            int n = line.length();
            char file[n + 1];
            strcpy(file, line.c_str());
            depth2pcd_normal(file);
        }
        myfile.close();
    }
    printf("The data is processed. End the application.\n");
    return 0;
}
