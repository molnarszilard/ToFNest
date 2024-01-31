

#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <ctime>

void downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr &points, float leaf_size,
                pcl::PointCloud<pcl::PointXYZ>::Ptr &downsampled_out)
{
  pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
  vox_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
  vox_grid.setInputCloud(points);
  vox_grid.filter(*downsampled_out);
}

void compute_surface_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr &points, float normal_radius,
                             pcl::PointCloud<pcl::Normal>::Ptr &normals_out)
{
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> norm_est;
  const clock_t begin_time = clock();

  // Use a FLANN-based KdTree to perform neighborhood searches
  norm_est.setSearchMethod(pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>));

  // Specify the size of the local neighborhood to use when computing the surface normals
  norm_est.setRadiusSearch(normal_radius);

  // Set the input points
  norm_est.setInputCloud(points);

  // Estimate the surface normals and store the result in "normals_out"
  norm_est.compute(*normals_out);
  std::cout << "time:" << float(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;
}

void visualize_normals(const pcl::PointCloud<pcl::PointXYZ>::Ptr points,
                       const pcl::PointCloud<pcl::PointXYZ>::Ptr normal_points,
                       const pcl::PointCloud<pcl::Normal>::Ptr normals)
{
  // Add the points and normals to the vizualizer
  pcl::visualization::PCLVisualizer viz;
  viz.addPointCloud(points, "points");
  viz.addPointCloud(normal_points, "normal_points");

  viz.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(normal_points, normals, 1, 0.01, "normals");

  // Give control over to the visualizer
  viz.spin();
}

int main(int argc, char **argv)
{
  // Load data from pcd

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  std::string filename = "";
  std::cout << "Which .pcd file do you want to compute the normals for?" << std::endl;
  std::cin >> filename;
  std::cout << "Processing: " << filename << std::endl;
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) //* load the file
  {
    PCL_ERROR("Couldn't read file \n");
    return (-1);
  }

  // Point Clouds to hold output
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

  const float normal_radius = 0.03;
  compute_surface_normals(cloud, normal_radius, normals);
  filename = filename.substr(0, filename.size() - 4);
  pcl::PointCloud<pcl::PointNormal> cloudnormals;
  pcl::PointNormal p;
  for (int i = 0; i < cloud->size(); i++)
  {
    p.x = cloud->points[i].x;
    p.y = cloud->points[i].y;
    p.z = cloud->points[i].z;
    p.normal_x = normals->points[i].normal_x;
    p.normal_y = normals->points[i].normal_y;
    p.normal_z = normals->points[i].normal_z;
    cloudnormals.points.push_back(p);
  }
  cloudnormals.width = cloudnormals.points.size();
  cout << "Number of points:" << cloudnormals.width << endl;
  cloudnormals.height = 1;
  cloudnormals.points.resize(cloudnormals.width * cloudnormals.height);
  cloudnormals.is_dense = false;
  pcl::io::savePCDFile(filename + "_pclnormals.pcd", cloudnormals, true);
  cout << "[*] Conversion finished!" << endl;

  return (0);
}
