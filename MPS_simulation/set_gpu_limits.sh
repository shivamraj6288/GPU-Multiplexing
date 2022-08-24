#!/bin/bash
if [ $# != 2 ];
then 
echo "invalide arguments passed"
exit 1
fi

if [ $1 != -1 ];
then 
export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=$1
else 
export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=24G
fi 

if [ $2 != -1 ];
then 
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$2
else 
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
fi 
