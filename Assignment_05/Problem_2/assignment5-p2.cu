// Compile: nvcc -g -G -arch=sm_52 -std=c++11 assignment5-p2.cu -o assignment5-p2

#include <cuda.h>
#include <iostream>
#include <sys/time.h>

#define THRESHOLD (0.000001)
//Set tile dimension to be a power of 2
#define tileDim (1<<11)

using std::cerr;
using std::cout;
using std::endl;


__host__ void host_excl_prefix_sum(float* h_A, float* h_O, int N) {
  h_O[0] = 0;
  for (int i = 1; i < N; i++) {
    h_O[i] = h_O[i - 1] + h_A[i - 1];
  }
}

__global__ void kernel_excl_prefix_sum(float* d_in, float* d_out, int N) {
  // TODO: Fill in

  int istart = (blockDim.x*blockIdx.x + threadIdx.x)*tileDim;
  int iend = min(N,(istart+tileDim));

  for(int i=istart+1;i<iend;i++){
      d_out[i] = d_out[i-1] + d_in[i-1];
  }

}

__global__ void kernel_preprocess(float* d_in, float* d_out, int N) {
  // TODO: Fill in

  int i = (blockDim.x*blockIdx.x + threadIdx.x)*tileDim;

  if(i>0 && i<N){
    for(int j=i-tileDim;j<i;j++){
      d_out[i] = d_out[i] + d_in[j];
    }
  }

}

__host__ void check_result(float* w_ref, float* w_opt, int N) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++) {
    this_diff = w_ref[i] - w_opt[i];
    if (fabs(this_diff) > THRESHOLD) {
      numdiffs++;
      if (this_diff > maxdiff)
        maxdiff = this_diff;
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over threshold " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

__host__ double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main() {
  const int N = (1 << 24);
  size_t size = N * sizeof(float);

  float* h_in = (float*)malloc(size);
  std::fill_n(h_in, N, 1);

  float* h_excl_sum_out = (float*)malloc(size);
  std::fill_n(h_excl_sum_out, N, 0);

  double clkbegin = rtclock();
  host_excl_prefix_sum(h_in, h_excl_sum_out, N);
  double clkend = rtclock();
  double time = clkend - clkbegin; // seconds
  cout << "Serial time on CPU: " << time * 1000 << " msec" << endl;

  float* h_dev_result = (float*)malloc(size);
  std::fill_n(h_dev_result, N, 0);
  float* d_in;
  float* d_out;
  cudaError_t status;
  cudaEvent_t start, end;
  // TODO: Fill in

  status = cudaMalloc(&d_in,size);
  status = cudaMalloc(&d_out,size);
  if(status!=cudaSuccess){
    cout<<"Error in Cuda Malloc"<<endl;
  }

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  int numtiles = (int)ceil(((double)N)/((double)tileDim));
  int threadPerBlock = min(numtiles,1<<10);
  int numBlock = (int) ceil(((double)numtiles)/((double)threadPerBlock));

  dim3 GridD(numBlock,1,1);
  dim3 BlockD(threadPerBlock,1,1);

  kernel_preprocess<<<GridD,BlockD>>>(d_in,d_out,N);

  status = cudaMemcpy(h_dev_result, d_out, size, cudaMemcpyDeviceToHost);  

  for(int i=tileDim;i<N;i+=tileDim){
    h_dev_result[i] = h_dev_result[i] + h_dev_result[i-tileDim];
  }

  status = cudaMemcpy(d_out, h_dev_result, size, cudaMemcpyHostToDevice);

  kernel_excl_prefix_sum<<<GridD,BlockD>>>(d_in,d_out,N);

  status = cudaMemcpy(h_dev_result, d_out, size, cudaMemcpyDeviceToHost);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  check_result(h_excl_sum_out, h_dev_result, N);
  float k_time; // ms
  cudaEventElapsedTime(&k_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cout << "Kernel time on GPU: " << k_time << " msec" << endl;

  // Free device memory
  cudaFree(d_in);
  cudaFree(d_out);

  free(h_in);
  free(h_excl_sum_out);
  free(h_dev_result);


  return EXIT_SUCCESS;
}
