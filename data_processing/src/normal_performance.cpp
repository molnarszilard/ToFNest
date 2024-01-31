#include <boost/algorithm/string.hpp>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <string>

using namespace std;
string directory = "";
double compare(char line[])
{
    pcl::PointCloud<pcl::PointNormal>::Ptr cloudgt(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloudpred(new pcl::PointCloud<pcl::PointNormal>);
    char file_gt[200];
    char file_pred[200];
    char file_delta[200];
    sprintf(file_gt, "%spcd/normals/%s.pcd", directory.c_str(), line);
    sprintf(file_pred, "%spcdpred/%s.pcd", directory.c_str(), line);
    sprintf(file_delta, "%spcdpred_delta/%s.pcd", directory.c_str(), line);

    if (pcl::io::loadPCDFile<pcl::PointNormal>(file_gt, *cloudgt) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read file \n");
        return (-1);
    }

    if (pcl::io::loadPCDFile<pcl::PointNormal>(file_pred, *cloudpred) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read file \n");
        return (-1);
    }

    std::cout << "PCD_GT has: " << cloudgt->size() << " data points." << std::endl;     //*
    std::cout << "PCD_PRED has: " << cloudpred->size() << " data points." << std::endl; //*
    double quality;
    double cloud_quality = 0;
    int histo[100] = {0};
    pcl::PointXYZRGB p;
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    if (cloudgt->size() < cloudpred->size())
    {
        for (int i = 0; i < cloudgt->size(); i++)
        {
            Eigen::Vector3f normalgt;
            normalgt << cloudgt->points[i].normal_x, cloudgt->points[i].normal_y, cloudgt->points[i].normal_z;
            normalgt.normalize();
            Eigen::Vector3f normalpred;
            normalpred << cloudpred->points[i].normal_x, cloudpred->points[i].normal_y, cloudpred->points[i].normal_z;
            normalpred.normalize();
            quality = normalgt.dot(normalpred);
            double angle = acos(quality);
            bool nan = false;
            if (quality != quality || std::isnan(quality))
                nan = true;
            if (!nan)
            {
                if (quality < 0)
                {
                    quality = -quality;
                }
                cloud_quality = cloud_quality + quality;
                p.x = cloudgt->points[i].x;
                p.y = cloudgt->points[i].y;
                p.z = cloudgt->points[i].z;
                std::uint8_t r = 0, g = 0, b = 0;
                if (quality <= 0.89)
                {
                    r = 4335 / 7 - quality * 255 / 35;
                    g = quality * 255 / 9 - 255 * 80 / 9;
                    b = 0;
                }
                else
                {
                    r = 0;
                    g = 255 * 98 / 9 - quality * 255 / 9;
                    b = quality * 51 - 4845;
                }
                std::uint32_t rgb = ((std::uint32_t)r << 16 | (std::uint32_t)g << 8 | (std::uint32_t)b);
                p.rgb = *reinterpret_cast<float *>(&rgb);
                cloud.points.push_back(p);
            }
        }
        cloud_quality = cloud_quality / cloudgt->size();
    }
    else
    {
        for (int i = 0; i < cloudpred->size(); i++)
        {
            Eigen::Vector3f normalgt;
            normalgt << cloudgt->points[i].normal_x, cloudgt->points[i].normal_y, cloudgt->points[i].normal_z;
            normalgt.normalize();
            Eigen::Vector3f normalpred;
            normalpred << cloudpred->points[i].normal_x, cloudpred->points[i].normal_y, cloudpred->points[i].normal_z;
            normalpred.normalize();
            quality = normalgt.dot(normalpred);
            double angle = acos(quality);
            bool nan = false;
            if (quality != quality || std::isnan(quality))
                nan = true;
            if (!nan)
            {
                if (quality < 0)
                {
                    quality = -quality;
                }
                cloud_quality = cloud_quality + quality;
                p.x = cloudpred->points[i].x;
                p.y = cloudpred->points[i].y;
                p.z = cloudpred->points[i].z;
                std::uint8_t r = 0, g = 0, b = 0;
                if (quality <= 0.75)
                {
                    r = 765 - quality * 1020;
                    g = quality * 1020 - 510;
                }
                else
                {
                    b = quality * 1020 - 765;
                    g = 1020 - quality * 1020;
                }
                std::uint32_t rgb = ((std::uint32_t)r << 16 | (std::uint32_t)g << 8 | (std::uint32_t)b);
                p.rgb = *reinterpret_cast<float *>(&rgb);
                cloud.points.push_back(p);
            }
        }
        cloud_quality = cloud_quality / cloudpred->size();
    }
    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.points.resize(cloud.width * cloud.height);
    cloud.is_dense = false;
    pcl::io::savePCDFile(file_delta, cloud, true);
    return cloud_quality;
}

int main(int argc, char *argv[])
{

    double Quality = 0;
    int cnt = 0;
    int histogram[100] = {};
    std::cout << "Directory where you depth/ and predictions/ folder?" << std::endl;
    std::cin >> directory;
    ifstream myfile(directory + "filelist.txt");
    if (myfile.is_open())
    {
        string line;
        while (getline(myfile, line))
        {
            int n = line.length();
            char file[n + 1];
            strcpy(file, line.c_str());
            double q = compare(file);
            int ind = (int)(q * 100);
            if (ind >= 100)
                ind = 99;
            histogram[ind]++;
            printf("quality for %s pcd is: %f\n", file, q);
            Quality += q;
            cnt++;
        }
        myfile.close();
    }
    double Q = Quality / cnt;
    printf("The TOTAL quality for %d pointclouds is: %f\n\n", cnt, Q);
    printf("The histogram:\n");
    ofstream file_h;
    file_h.open("histogram.txt");
    for (int i = 0; i < 100; i++)
    {
        if (histogram[i] > 0)
        {
            file_h << i << " " << histogram[i] << "\n";
            printf("%02d:%04d - ", i, histogram[i]);
            for (int j = 0; j < histogram[i]; j++)
                printf("|");
            printf("\n");
        }
    }
    file_h.close();
    return 0;
}
