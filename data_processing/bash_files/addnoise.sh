#!bin/bash
cd $1"depth/"

for filename in *.png; do
    cd "$2"
    echo "$1depth/$filename" | ./addnoise2depth
done
mkdir -p $1"depth_noise/"
cd $1"depth/"
for filename in *noise.png; do
    mv "$filename" "../depth_noise/$filename"
done
cd "$2../"
python3 augmentation.py --dir="$1depth_noise/"