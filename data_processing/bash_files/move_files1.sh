#!bin/bash
cd $1
mkdir -p "pcd"
mkdir -p "depth3"
cd "depth"
for filename in *.pcd; do
    file="../pcd/"$filename
    mv "$filename" "$file"
done
for filename in *_d3.png; do
    file="../depth3/"$filename
    mv "$filename" "$file"
done