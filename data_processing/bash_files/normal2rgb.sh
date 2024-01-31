#!bin/bash
cd $1
for filename in *normals.pcd; do
    cd "$2"
    echo "$1$filename" | ./normal2rgb
done
