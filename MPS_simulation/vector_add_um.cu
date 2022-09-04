#include <cuda.h>
#include <bits/stdc++.h>
#include <unistd.h>
#include <ctime>
using namespace std;

ofstream log_file;
time_t start_p;
int rep=1000000;
long long rem =10000;

int num_slice = 4;
int slice_size;
int slice_no=0;

void custom_log(time_t time, long long n) {
    log_file<<difftime(time,start_p)<<"," <<n << endl;
}

void vec_init(int *a, int n) {
    for (int i=0; i<n; i++) {
        a[i]=rand()%1000;
    }
}
void print (int *a, int n) {
    for (int i=0; i<n; i++) {
        cout << a[i] << " ";
        
    }
    cout << endl;
}

__global__ void vec_add(int *a, int *b, int *c, long long n, int slice_size, int slice_no ) {
    long long id = threadIdx.x+blockIdx.x*blockDim.x;
    
    id = id + slice_size * slice_no *1024;
    // for (int i=id; i<n; i+=n/4){
    //     c[id]=a[id]+b[id];
    // }
    c[id]=a[id]+b[id];
}

bool check(const int *a, const int *b, const int *c, long long n){
    for (long long i=0; i<n; i++) {
        if (a[i]+b[i]!=c[i]){
            cout << "FAILED" <<endl;
            cout << "error at index = " <<i << endl;
            return false;
        }
    }
    cout << "Successfull"<<endl;
    return true;
}


int main (int argc, char* argv[]) {
    if(argc>1) {
        rem = atoi(argv[1]);
    }
    time(&start_p);
    int id;
    cudaGetDevice(&id);
    long long n;
    n= 1<<26;
    size_t bytes = n*sizeof(int);
    int *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    vec_init(a,n);
    vec_init(b,n);
    vec_init(c,n);

    cudaMemPrefetchAsync(a,bytes,id);
    cudaMemPrefetchAsync(b,bytes,id);

    long long block_size = 1024;
    long long grid_size = (n + block_size - 1) / block_size;

    time_t ct, st;
    time(&st);
    long long through_put=0;
    cout << "Process Starting" << endl;
    cout << "Process PID " << getpid() << endl;
    cout << "Start Time " << st << endl;
    string csv_file_name = "log"+to_string(getpid())+".csv";
    log_file.open(csv_file_name);

    
    slice_size = grid_size/num_slice;
    int new_grid_size = slice_size;
    int *thread_id;
    int *d_c;
    // cudaMalloc(&d_c, bytes);
    for(int  i=0; i<num_slice; i++) {
        // cudaMemcpy(d_c, c, bytes, cudaMemcpyHostToDevice);
        vec_add<<<new_grid_size, block_size>>> (a,b,c,n, slice_size, i);
        cudaMemPrefetchAsync(c,bytes,cudaCpuDeviceId);
        cudaDeviceSynchronize();
        check(a,b,c,n);
    }
    
    
    // while(rem--){
    //     vec_add<<<grid_size, block_size>>> (a,b,c,n);
    //     through_put++;
    //     // cout << through_put << endl;
    //     time (&ct);
    //     if(difftime(ct,st)>=5 || rem==0){
    //         custom_log(ct,through_put/5);
    //         cout << "Thorughput " << through_put/5 <<endl;
    //         cout << "Remaining " << rem << endl;
    //         st=ct;
    //         through_put=0;
    //     }
    // }
    // sleep(2);
    // cudaDeviceSynchronize();
    // cudaMemPrefetchAsync(c,bytes,cudaCpuDeviceId);

    // check(a,b,c,n);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    log_file.close();


    return 0;
}