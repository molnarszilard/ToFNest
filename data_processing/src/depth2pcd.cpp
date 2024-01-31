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
#include <string>

using namespace std;

int main(int argc, char **argv)
{
    int min_points=3;
    std::string filename = "";
    std::cout << "Which .pcd file do you want to compute the normals for?" << std::endl;
    std::cin >> filename;
    std::cout << "Processing: " << filename << std::endl;
    cv::Mat depth = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
    
    // MODIFY THE K matrix according to your camera, here is an example with a pico sense camera
    double K[9] = {460.58518931365654, 0.0, 334.0805877590529, 0.0, 460.2679961517268, 169.80766383231037, 0.0, 0.0, 1.0};
    double fx = K[0];
    double fy = K[4];
    double x0 = K[2];
    double y0 = K[5];

    pcl::PointCloud<pcl::PointXYZ> cloud_msg;
    pcl::PointXYZ p;
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

            cloud_msg.points.push_back(p);
        }
    }
    cloud_msg.width = cloud_msg.points.size();
    cout << "Number of points:" << cloud_msg.width << endl;
    if (cloud_msg.width > min_points)
    {
        cloud_msg.height = 1;
        cloud_msg.points.resize(cloud_msg.width * cloud_msg.height);
        cloud_msg.is_dense = false;
        filename = filename.substr(0, filename.size() - 3);
        pcl::io::savePCDFile(filename+"pcd", cloud_msg, true);
        cout << "[*] Conversion finished!" << endl;
    }
    else
    {
        cout << "[*] Conversion failed - too few points!" << endl;
    }
    return 0;
}
