#!/bin/bash
reps=10000
echo "Compiling file"
nvcc -o vec.o vector_add_um.cu
echo "File compiled"
echo "Running P1 independently"
./vec.o $reps &
pid1=$!
wait 
echo "P1 complete"
echo "Running P1"
./vec.o $reps &
pid2=$!
for i in {1..6}
do 
rem=$(( 6 - i ))
echo "Waiting for $rem secs";
sleep 1 
done
echo "Running P2"
./vec.o $reps &
pid3=$!
wait 
echo "Code executed. Analysing results..."
python3 analysis.py $pid1 $pid2 $pid3 
# for i in {6..10}
# do 
# rem=$(( 15 - i ))
# echo "Time remaining $rem secs";
# sleep(1)
# done