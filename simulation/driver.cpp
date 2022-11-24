#include <cuda.h>
#include <bits/stdc++.h>
#include <unistd.h>

#include "controller_class.h"
using namespace std;

CUmodule module;
CUfunction function;

int Controller::available_blocks=0;
bool Controller::scheduling_process_running=false;
queue<QueueElements> Controller::pending_queue;

int main(int *argc, char *argv[]) {
    const char* ptx_file = "vec_add_kernel.ptx";
    // const char* kernel_name = ""
    Controller controller;
    // controller.available_blocks=0;
    // controller.scheduling_process_running=false;
    controller.start();

}