#!bin/bash
cd $1
for filename in *.png; do
    cd "$2"
    echo "$1$filename" | ./depth3
done