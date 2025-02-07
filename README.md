# ToFNest: Efficient  normal  estimation  for  ToF Depth  cameras

## Overview
In this work we propose an efficient normal estimation method for  depth  images  acquired  by  Time-of-Flight  (ToF)  cameras based  on  feature  pyramid  networks.  We  do  the  our  normal estimation training starting from the 2D depth images, projecting  the  measured  data  into  the  3D  space  and  computing  theloss  function  for  the  pointcloud  normal.  Despite  the  simplicity of the methods it proves to be efficient in terms of robustness.Compared with the state of the art methods, our method proved to be faster with similar precision metrics from other methods on  public  datasets.  In order  to  validate  our  proposed  solution we  made  an  extensive  testing  using  both  public  datasets  and custom recorded indoor and ourdoor datasets as well.

This work was published on [ICCV2021](https://www.researchgate.net/publication/356511911_ToFNest_Efficient_normal_estimation_for_time-of-flight_depth_cameras) and [Sensors2021](https://www.researchgate.net/publication/354758865_Feature_Pyramid_Network_Based_Efficient_Normal_Estimation_and_Filtering_for_Time-of-Flight_Depth_Cameras).

## Content
- [Prerequisites](#prerequisites)
- [Create conda env](#create-conda-env)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Real-time evaluation](#real-time-evaluation)
- [Demo](#demo)

## Prerequisites
The code was built using the following libraries ([environment.yml](environment.yml)):
- [Python](https://www.python.org/downloads/)  >= 3.6
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.3
- [Scipy](https://github.com/scipy/scipy)
- [OpenCV](https://github.com/opencv/opencv)
- [Matplotlib](https://matplotlib.org/stable/index.html)

### Create conda env

You can create it automatically:

```conda env create -f environment.yml```

or manually:

```conda create -n tofnest python=3.6```

you can change the version of pytorch. Tested with 1.4 and 1.8

```conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia -y```

```conda install -c anaconda scipy -y```

```conda install -c conda-forge matplotlib -y```

```conda install -c conda-forge opencv -y```

```conda install -c anaconda pillow```

additionally install with pip: numpy, timeit

## [Data Preparation](data_processing)

You might need to have PointCloudLibrary and opencv installed.

```mkdir build```

```cd build```

```cmake ..```

```make```

### Dataset preparation

If you have a rosbag file, you can sample depth images (also IR and RGB images, PCD-s or a combination of IR and Depth) from a ROS bag file.
By default only the depth images are saved, if you want the rest comment out the correspondong lines in the .cpp file.

Create a catkin_workspace, copy this repo inside the src. run ```catkin_make``` then ```source devel/setup.bash```

To run the code, first start the rosmaster (```roscore```), then run ```roslaunch save_depth_images save_depth_images.launch```
The code will wait for you to play the bag file (```rosbag play rosbag.bag```)

After you have obtained your depth images, and they are inside path/to/dataset/depth go into bash_files/ and run:

```bash create_dataset.sh path/to/dataset/ path/to/build/```

Initially the normals are computed using PCLNormals, but you can use your own preferred version (just modify the code accordingly).
The data for training will be stored in depth3/ and in normalimages/ as the ground truth.

### Data augmentation
If you want to apply some augmentations on your depth images run:
python3 augmentation.py --dir=path/to/dataset/depth/
And run the Dataset preparation for your augmented data.

Add noise to your data by running 

```bash addnoise.sh path/to/dataset/ path/to/build/```

you can add noise to your depth images. (if you want to set the noise level, modify the src/addnoise2depth.cpp file), and then run the code, from Dataset preparation (note you might want to calculate your the normals for your data without noise, and in this case it is enough to create the depth3 images, be sure that all the names are correct)



## Training and Evaluation
![arch_new](https://user-images.githubusercontent.com/22835687/109430618-f692a780-7a0a-11eb-9270-1421457f8433.png)

### Training

Modify in datasetloader.py the path to the folder containing your depth images (16bit 1 or 3 channel) and the images about the normal vector (8bit RGB). The depth images can be simple depth images, in order to increase the dataloading speed you might want to create 3 channel depth images (the same data from the depth image is copied), or you can also experiment with different combinations, like 2 channel containing depth information, and 1 channel containing a monochrome image (all of these should be on 16 bit).
In train.py you can see the available options, and modify them, either from code, or using them as [--options] at running. You might want to set the code to save images from the training phase, so you can see the evolution of the training.

Run python train.py [--options], example (data_dir is for the dataset directory. depth_dir is the folder containing depth images splitted into train and test folders. You can set the number of epochs during training, the batch size, learning rate, etc... see train.py for further options):

```python train.py --data_dir=./dataset/ --depth_dir=depth3 --epochs=10 --bs=1```

### Evaluation

Here you can create a prediction on a single image, then set the path to that image, or you can predict the normal images for an entire folder, by adding the --eval_folder=True flag in addition to the folder path.

Run python eval.py [--options], for example (depth_folder can be a file or an entire folder):

```python eval.py --depth_folder=./dataset/depth3/test/ --model_path=./saved_models/d2n_1_9.pth --pred_folder=./pred_images/```


If you have created your normaimages (they should look like something like this: 00000_pred.png), copy them to path/to/dataset/predictions/

run

```bash depth2pcd_normal.sh path/to/dataset/ path/to/build/```

```bash normal_performance.sh```

in path/to/build/histogram.txt you can find the histogram about your model
in path/to/dataset/pcdpred_delta/ there are the pointclouds that show you which part of the pcd was predicted correctly or not.

You can use your own dataset, in this case you have to create your own normal ground truth using a preferred method for this.
For training you will need the depth images with their corresponding image containing the normal vector coordinates decoded in rgb values.

You can also download the NYU_V2 (https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset, that contains the depth images and the gt normals (https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/configs/benchmark_tasks/surface_normal_estimation/README.md or  https://cs.nyu.edu/~deigen/dnl/normals_gt.tgz). This might need some additional steps to read the files.

### Real-time evaluation

This code can be run using depth images from a ROS topic.
A bag file (~5GB), that contains the depth topic (/pico_zense/depth/), can be found at https://drive.google.com/file/d/1aQ2M3FsPrUHNpcYEZz-6i8YmFC5fWTEB/view?usp=sharing


You need to install ROS Melodic. 

inside conda environment:
```pip install pyyaml```
```pip install rospkg```
```pip install rospy```

Input_topic is the base topic, this means that, this topic has two other subtopics: image_raw, and camera_info. You have to specify the base topic as argument, or modify the .py file or your bag is differently structured.

Then start a roscore, and the bag (```rosbag play -l rosbag.bag``` (-l means it is played in a loop, repeatedly). (does not require conda))

```python tofnest_rt.py``` (requires conda)

By default only the normal images are published on /normal_image topic. If you want to puv=blish the colored pcd, set publish_pcd flag to True. (Note that this will drastically reduce the speed, since the conversion between a depth image to a pcd can be a lenghty process, especially in python).

## ADI Smart Camera
Setup aditof camera for tofnest

Find the ip of camera, make sure that you can connect to it with ssh

After you've connected to the camera:
- **On camera:**
  - `cd Workspace/aditof_sdk/build`
  - `sudo ./apps/server/aditof-server`
- **On your machine**
  - The following lines can be found (more detailed) here:
    - https://github.com/analogdevicesinc/aditof_sdk/blob/master/doc/linux/build_instructions.md
    - https://github.com/analogdevicesinc/aditof_sdk/blob/master/doc/3dsmartcam1/build_instructions.md
    - https://github.com/analogdevicesinc/aditof_sdk/tree/master/bindings/ros
  - **First time only in a _workspace_ folder:**
    - `cd workspace`
    - **Glog**
    - `git clone --branch v0.3.5 --depth 1 https://github.com/google/glog`
    - `cd glog`
    - `mkdir build_0_3_5 && cd build_0_3_5`
    - `cmake -DWITH_GFLAGS=off -DCMAKE_INSTALL_PREFIX=/opt/glog ..`
    - `sudo make -j4 && sudo make install`
    - `cd ../..`
    - **Libwebsockets**
    - `sudo apt-get install libssl-dev`
    - `git clone --branch v3.2.3 --depth 1 https://github.com/warmcat/libwebsockets`
    - `cd libwebsockets`
    - `mkdir build_3_2_3 && cd build_3_2_3`
    - `cmake -DLWS_STATIC_PIC=ON -DCMAKE_INSTALL_PREFIX=/opt/websockets ..`
    - `sudo make -j4 && sudo make install`
    - `cd ../..`
    - **Protobuf**
    - `git clone --branch v3.9.0 --depth 1 https://github.com/protocolbuffers/protobuf`
    - `cd protobuf`
    - `mkdir build_3_9_0 && cd build_3_9_0`
    - `cmake -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=/opt/protobuf ../cmake`
    - `sudo make -j4 && sudo make install`
    - `cd ../..`
    - **ADIToF SDK**
    - `git clone https://github.com/analogdevicesinc/aditof_sdk`
    - `cd aditof_sdk`
    - `mkdir build && cd build`
    - `cmake -DWITH_EXAMPLES=off -DWITH_ROS=on -DCMAKE_PREFIX_PATH="/opt/glog;/opt/protobuf;/opt/websockets" ..`
    - `sudo cmake --build . --target install`
    - `cmake --build . --target aditof_ros_package`
  - **Usage**
    - I recommend you taking a look here: https://github.com/analogdevicesinc/aditof_sdk/tree/master/bindings/ros#usage 
    - `cd workspace/aditof_sdk/build/catkin_ws`
    - `source devel/setup.bash`
    - `roslaunch aditof_roscpp camera_node.launch ip:="your.cam.era.ip"`
    - **Or as node:** 
    - `roscore`
    - `rosrun aditof_roscpp aditof_camera_node your.cam.era.ip`
  - ** WARNING: PROBLEMS may appear if the two aditof_sdk packages are not the same version (both on camera and on your machine).**
  - **After that you should be able to see all the camera topics in any terminal from your machine. Good luck!**

Change the topic parameters in the tofnest_rt.py file if necessary.

## Demo

Full video are available at: https://youtu.be/cOSoMvRneVw


![ezgif com-gif-maker](https://user-images.githubusercontent.com/22835687/109798142-0f75a580-7c23-11eb-9d65-3dff8d8f3439.gif)
## Citations
### Citing this work
If you find our code / paper / data useful to your research, please consider citing:

```bibtex
@InProceedings{molnar2021ToFNestEfficientNormal,
  author    = {Szil{\'{a}}rd Moln{\'{a}}r and Benjamin Kel{\'{e}}nyi and Levente Tam{\'{a}}s},
  booktitle = {{{IEEE/CVF} International Conference on Computer Vision Workshops, {ICCVW} 2021, Montreal, BC, Canada, October 11-17, 2021}},
  title     = {{ToFNest: Efficient Normal Estimation for Time-of-Flight Depth Cameras}},
  year      = {2021},
  pages     = {1791--1798},
  publisher = {{IEEE}},
  doi       = {10.1109/ICCVW54120.2021.00205},
}

@Article{molnar2021FeaturePyramidNetwork,
  author  = {Szil{\'{a}}rd Moln{\'{a}}r and Benjamin Kel{\'{e}}nyi and Levente Tamas},
  journal = {Sensors},
  title   = {{Feature Pyramid Network Based Efficient Normal Estimation and Filtering for Time-of-Flight Depth Cameras}},
  year    = {2021},
  number  = {18},
  pages   = {6257},
  volume  = {21},
  doi     = {10.3390/S21186257},
}
```


The main file types are point clouds, RGB images and depth images captured by Pico Zense cameras and ADI Smart Cameras (Courtesy of Analog Devices).

