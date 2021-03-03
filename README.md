# ToFNest: Efficient  normal  estimation  for  ToF Depth  cameras

## Abstract
In this work we propose an efficient normal estimation method for  depth  images  acquired  by  Time-of-Flight  (ToF)  cameras based  on  feature  pyramid  networks.  We  do  the  our  normal estimation training starting from the 2D depth images, projecting  the  measured  data  into  the  3D  space  and  computing  theloss  function  for  the  pointcloud  normal.  Despite  the  simplicity of the methods it proves to be efficient in terms of robustness.Compared with the state of the art methods, our method proved to be faster with similar precision metrics from other methods on  public  datasets.  In order  to  validate  our  proposed  solution we  made  an  extensive  testing  using  both  public  datasets  and custom recorded indoor and ourdoor datasets as well.

## Overview

## Content
- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [How to run](#how-to-run)
- [Demo](#demo)

## Prerequisites
The code was built using the following libraries ([requirements.txt](requirements.txt)):
- [Python](https://www.python.org/downloads/)  >= 3.6
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.3
- [Scipy](https://github.com/scipy/scipy)
- [OpenCV](https://github.com/opencv/opencv)
- [Imageio](https://imageio.github.io/)
- [Matplotlib](https://matplotlib.org/stable/index.html)
- [Constants](https://pypi.org/project/constants/)

## Data Preparation

You may use this https://github.com/molnarszilard/ToFNest_data_processing repo to create training dataset, or to evaluate your model.

You can use your own dataset, in this case you have to create your own normal ground truth using a preferred method for this.
For training you will need the depth images with their corresponding image containing the normal vector coordinates decoded in rgb values.

You can also download the NYU_V2 (https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset, that contains the depth images and the gt normals (https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/configs/benchmark_tasks/surface_normal_estimation/README.md or  https://cs.nyu.edu/~deigen/dnl/normals_gt.tgz). This might need some additional steps to read the files.

## Training and Evaluation
![arch_new](https://user-images.githubusercontent.com/22835687/109430618-f692a780-7a0a-11eb-9270-1421457f8433.png)

## How to run

### Training

Modify in datasetloader.py the path to the folder containing your depth images (16bit 1 or 3 channel) and the images about the normla vector (8bit RGB). The depth images can be simple depth images, in order to increase the dataloading speed you might want to create 3 channel depth images (the same data from the depth image is copied), or you can also experiment with different combinations, like 2 channel containing depth information, and 1 channel containing a monochrome image (all of these should be on 16 bit).
In train.py you can see the available options, and modify them, either from code, or using them as [--options] at running. You might want to set the code to save images from the training phase, so you can see the evolution of the training.

Run python train.py [--options]

### Evaluation

Here you can create a prediction on a single image, then set the path to that image, or you can predict the normal images for an entire folder, by adding the --eval_folder=True flag in addition to the folder path.

Run python eval.py [--options]

At https://github.com/molnarszilard/ToFNest_data_processing, you can find a code that compares the GT pointcloud with normals to your generated normal images.
## Demo

Full video are available at: TODO SZILARD -> YOUTUBE LINK


![ezgif com-gif-maker](https://user-images.githubusercontent.com/22835687/109798142-0f75a580-7c23-11eb-9d65-3dff8d8f3439.gif)



