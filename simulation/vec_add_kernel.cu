extern "C" __global__ void vecadd_kernel(const float* A, const float* B, float* C, int N) {
    int id = blockDim.x + blockIdx.x + threadIdx.x;
    if(id<N) C[id] = A[id]+B[id];
}