#!bin/bash
cd $1
for filename in *n2rgb.pcd; do
    cd "$2"
    echo "$1$filename" | ./cloud2image
done
