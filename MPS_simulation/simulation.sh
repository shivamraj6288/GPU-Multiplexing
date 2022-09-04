#!/bin/bash
reps=100000
resource_per_process=20

# ===============compiling============ #
echo "Compiling file"
nvcc -o vec.o vector_add_um.cu
echo "File compiled"

./set_gpu_limits.sh -1 -1


echo "Running P1 independently"
./vec.o $reps &
pid1=$!
echo "Process $pid1 starts" > "log.txt"
wait 
echo "P1 complete"
echo "Process $pid1 completes" >> "log.txt"
echo "Running P1"
./vec.o $reps &
pid2=$!
echo "Process $pid2 starts" >> "log.txt"
for i in {1..10}
do 
rem=$(( 11 - i ))
echo "Waiting for $rem secs";
sleep 1 
done
echo "Running P2"
./vec.o $reps &
pid3=$!
echo "Process $pid3 starts" >> "log.txt"
wait 

#===================== setting gpu limit ==============
echo "Setting resource limit"
echo "Setting resource limit" >> log.txt

./set_gpu_limits.sh -1 $resource_per_process

echo "Running P1 independently"
./vec.o $reps &
pid4=$!
echo "Process $pid4 starts" >> "log.txt"
wait 
echo "P1 complete"
echo "Process $pid4 completes" >> "log.txt"
echo "Running P1"
./vec.o $reps &
pid5=$!
echo "Process $pid5 starts" >> "log.txt"
for i in {1..10}
do 
rem=$(( 11 - i ))
echo "Waiting for $rem secs"
sleep 1 
done
echo "Running P2"
./vec.o $reps &
pid6=$!
echo "Process $pid6 starts" >> "log.txt"
wait 

# =============================
echo "Code executed. Analysing results..."
python3 analysis.py $pid1 $pid2 $pid3 $pid4 $pid5 $pid6



