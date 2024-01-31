#!bin/bash

cd $1
echo $pwd

mkdir -p "normals"
for filename in *normals.pcd; do
    file="normals/"$filename
    mv "$filename" "$file"
done

mkdir -p "n2rgb"
for filename in *n2rgb.pcd; do
    file="n2rgb/"$filename
    mv "$filename" "$file"
done

mkdir -p "normalimages"
for filename in *rgb.png; do
    file="normalimages/"$filename
    mv "$filename" "$file"
done

mkdir -p "pcd"
for filename in *.pcd; do
    file="pcd/"$filename
    mv "$filename" "$file"
done

mv normalimages/ ../normalimages/