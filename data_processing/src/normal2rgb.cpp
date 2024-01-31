#include <iostream>
#include <string>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
using namespace std;

int main(int argc, char *argv[])
{
    bool normalize = true;
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    pcl::PointXYZRGB p;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud0(new pcl::PointCloud<pcl::PointNormal>);
    std::string filename = "";
    std::cout << "Input pcd file with x,y,z,nx,ny,nz parameters" << endl;
    std::cin >> filename;
    printf("Processing: %s\n",filename);
    if (pcl::io::loadPCDFile<pcl::PointNormal>(filename, *cloud0) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read file \n");
        return (-1);
    }
    filename = filename.substr(0, filename.size() - 4);
    std::cout << "PointCloud has: " << cloud0->size() << " data points." << std::endl; //*

    if (normalize)
    {
        for (int i = 0; i < cloud0->size(); i++)
        {
            p.x = cloud0->points[i].x;
            p.y = cloud0->points[i].y;
            p.z = cloud0->points[i].z;
            double r0 = cloud0->points[i].normal_x;
            double g0 = cloud0->points[i].normal_y;
            double b0 = cloud0->points[i].normal_z;
            std::uint8_t r = 0, g = 0, b = 0;
            Eigen::Vector3f vector;
            vector[0] = r0;
            vector[1] = g0;
            vector[2] = b0;
            vector.normalize();
            vector[0] = (vector[0]+1)/2;
            vector[1] = (vector[1]+1)/2;
            vector[2] = (vector[2]+1)/2;
            r = (int)(vector[0] * 255);
            g = (int)(vector[1] * 255);
            b = (int)(vector[2] * 255);
            std::uint32_t rgb = ((std::uint32_t)r << 16 | (std::uint32_t)g << 8 | (std::uint32_t)b);
            p.rgb = *reinterpret_cast<float *>(&rgb);

            cloud.points.push_back(p);
        }
    }
    else
    {
        double minr = 0, maxr = 0;
        double ming = 0, maxg = 0;
        double minb = 0, maxb = 0;
        for (int i = 0; i < cloud0->size(); i++)
        {
            double r0 = cloud0->points[i].normal_x;
            double g0 = cloud0->points[i].normal_y;
            double b0 = cloud0->points[i].normal_z;
            if (minr > r0)
                minr = r0;
            if (ming > g0)
                ming = g0;
            if (minb > b0)
                minb = b0;
            if (maxr < r0)
                maxr = r0;
            if (maxg < g0)
                maxg = g0;
            if (maxb < b0)
                maxb = b0;
        }
        for (int i = 0; i < cloud0->size(); i++)
        {
            p.x = cloud0->points[i].x;
            p.y = cloud0->points[i].y;
            p.z = cloud0->points[i].z;
            double r0 = cloud0->points[i].normal_x;
            double g0 = cloud0->points[i].normal_y;
            double b0 = cloud0->points[i].normal_z;
            std::uint8_t r = 0, g = 0, b = 0;
            r = (int)((r0 - minr) / (maxr - minr) * 255);
            g = (int)((g0 - ming) / (maxg - ming) * 255);
            b = (int)((b0 - minb) / (maxb - minb) * 255);
            std::uint32_t rgb = ((std::uint32_t)r << 16 | (std::uint32_t)g << 8 | (std::uint32_t)b);
            p.rgb = *reinterpret_cast<float *>(&rgb);

            cloud.points.push_back(p);
        }
    }

    cloud.width = cloud.points.size();
    cout << "Number of points:" << cloud.width << endl;
    cloud.height = 1;
    cloud.points.resize(cloud.width * cloud.height);
    cloud.is_dense = false;
    pcl::io::savePCDFile(filename + "_n2rgb.pcd", cloud, true);

    cout << "[*] Conversion finished!" << endl;
    return 0;
}
