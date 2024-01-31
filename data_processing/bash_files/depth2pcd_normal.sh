#!bin/bash

python3 list_files.py --dir="$1depth/"
cd $1
mkdir -p "pcdpred"
cd "$1depth/"
mv filelist.txt ../filelist.txt
cd $2
echo $1 | ./depth2pcd_normal

