#!bin/bash

bash depth2pcd.sh "$1depth/" "$2"
bash depth3.sh "$1depth/" "$2"
bash move_files1.sh "$1"
bash sweep.sh "$1pcd/" "$2"
bash compute_normals.sh "$1pcd/" "$2"
bash normal2rgb.sh "$1pcd/" "$2"
bash cloud2image.sh "$1pcd/" "$2"
bash move_files2.sh "$1pcd/"
python3 rename.py --dir="$1depth/"
python3 rename.py --dir="$1depth3/"
python3 rename.py --dir="$1normalimages/"
python3 rename.py --dir="$1pcd/normals/" --extension=".pcd"