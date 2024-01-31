#!bin/bash
cd $1
mkdir -p "pcdpred_delta"
cd $2
echo $1 | ./normal_performance
