#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <iostream>
#include <thread>
#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/centroid.h>
#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/surface/gp3.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/vtk_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl_msgs/PolygonMesh.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <pcl/io/io.h>
#include <string>
#include <math.h>
#include <visualization_msgs/Marker.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace sensor_msgs;
using namespace message_filters;

typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point> PointCloud;
//typedef sensor_msgs::PointCloud2 PointCloud;
typedef pcl::PointXYZ PointT;
typedef sync_policies::ApproximateTime<Image, Image, PointCloud, Image> MySyncPolicy;

unsigned int cnt = 0;
char directory[100] = "./dataset/normalrgb_files/";

void callback(const ImageConstPtr &ir, const ImageConstPtr &depth, const PointCloud::ConstPtr &cloud_in, const ImageConstPtr &rgb)
{
    //ROS_INFO_STREAM("Data Arrival\n");

    // cv_bridge::CvImagePtr img_ptr_ir;
    // cv_bridge::CvImagePtr img_ptr_rgb;
    cv_bridge::CvImagePtr img_ptr_depth;
    // std::string enci=ir->encoding;
    // std::cout<<enci<<std::endl;
    // img_ptr_ir = cv_bridge::toCvCopy(*ir, sensor_msgs::image_encodings::TYPE_16UC1);
    // img_ptr_rgb = cv_bridge::toCvCopy(*rgb, sensor_msgs::image_encodings::TYPE_8UC3);
    img_ptr_depth = cv_bridge::toCvCopy(*depth, sensor_msgs::image_encodings::TYPE_16UC1);
    // cv::Mat &mat_ir = img_ptr_ir->image;
    // cv::Mat mat_ir_contrast = cv::Mat::zeros(mat_ir.rows, mat_ir.cols, CV_16UC1);
    // cv::Mat &mat_rgb = img_ptr_rgb->image;
    cv::Mat &mat_depth = img_ptr_depth->image;
    // cv::convertScaleAbs(mat_ir, mat_ir_contrast, 0.15, 0.0);

    // cv::convertScaleAbs(mat_depth, mat_depth, 0.03, 1.0);
    // cv::Mat zerochannel = cv::Mat::zeros(cv::Size(mat_depth.rows, mat_depth.cols), CV_16U);
    // cv::Mat output = cv::Mat::zeros(mat_depth.rows, mat_depth.cols, CV_16UC3);
    // cv::Mat images[3] = {mat_ir, mat_depth, zerochannel};
    // int dims[3] = {2, mat_depth.rows, mat_depth.cols};
    // cv::Mat joined(3, dims, CV_16U);
    // for (int i = 0; i < 3; i++)
    // {
    //     uint16_t *ptr = &joined.at<uint16_t>(i, 0, 0);                            // pointer to first element of slice i
    //     cv::Mat destination(mat_depth.rows, mat_depth.cols, CV_32S, (void *)ptr); // no data copy, see documentation
    //     images[i].copyTo(destination);
    // }

    // for (int x = 0; x < images[0].rows; x++)
    // {
    //     for (int y = 0; y < images[0].cols; y++)
    //     {
    //         output.at<cv::Vec3s>(x, y)[2] = images[0].at<unsigned short>(x, y);
    //         output.at<cv::Vec3s>(x, y)[1] = images[1].at<unsigned short>(x, y);
    //         // output.at<cv::Vec3s>(x, y)[0] = images[2].at<unsigned short>(x, y);
    //     }
    // }

    // char file_ir[100];
    // // char file_ir_contrast[100];
    // char file_rgb[100];
    // char file_pcd[100];
    char file_depth[100];
    // sprintf(file_ir, "%s%05d_ir.png", directory, cnt);
    // // sprintf(file_ir_contrast, "%s%04d_ir_contrast.png", directory, cnt);
    // sprintf(file_rgb, "%s%05d_rgb.png", directory, cnt);
    // sprintf(file_pcd, "%s%05d_pcd.pcd", directory, cnt);
    sprintf(file_depth, "%s%05d_depth.png", directory, cnt);

    // cv::imwrite(file_ir, mat_ir);
    // // cv::imwrite(file_ir_contrast, mat_ir_contrast);
    // cv::imwrite(file_rgb, mat_rgb);
    cv::imwrite(file_depth, mat_depth);
    // pcl::io::savePCDFileASCII(file_pcd, *cloud_in);
    // std::cout << "Input data number " << int(cnt) << " is saved" << std::endl;
    // char file_depthir[100];
    // sprintf(file_depthir, "%s%04d_depthir.png", directory, cnt);
    // cv::imwrite(file_depthir, output);
    // std::cout << "Depth+ir is saved" << std::endl;
    cnt++;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "normalrgb");
    ros::NodeHandle nh_;
    // message_filters::Subscriber<Image> ir_sub(nh_, "ir_input", 1000);
    message_filters::Subscriber<Image> depth_sub(nh_, "depth_input", 1000);
    // message_filters::Subscriber<PointCloud> pcd_sub(nh_, "point_cloud_in", 1000);
    // message_filters::Subscriber<Image> rgb_sub(nh_, "rgb_input", 1000);
    // Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), ir_sub, depth_sub, pcd_sub, rgb_sub);
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), depth_sub);
    //TimeSynchronizer<Image, Image, PointCloud, Image> sync(ir_sub, depth_sub, pcd_sub, rgb_sub, 10);
    // sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4));
    sync.registerCallback(boost::bind(&callback, _1));
    ros::spin();
}
