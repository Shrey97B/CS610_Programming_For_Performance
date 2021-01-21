// Compile: nvcc -g -G -arch=sm_52 -std=c++11 assignment5-p5.cu -o assignment5-p5

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

const uint64_t N = (64);
#define THRESHOLD (0.000001)
//TileD should be less than or equal to 10 due to thread_per_blolck limit
#define TileD (10)

using std::cerr;
using std::cout;
using std::endl;

// TODO: Edit the function definition as required
__global__ void kernel1(float* d_in,float* d_out) {

  int i = (blockIdx.x*blockDim.x + threadIdx.x);
  int j = (blockIdx.y*blockDim.y + threadIdx.y);
  int k = (blockIdx.z*blockDim.z + threadIdx.z);

  if(i>0 && j>0 && k>0 && i<(N-1) && j<(N-1) && k<(N-1)){
  float temp = 0;
  temp = (d_in[(i-1)*N*N + j*N + k] + d_in[(i+1)*N*N + j*N + k]);
  temp += d_in[i*N*N + (j-1)*N + k];
  temp += d_in[i*N*N + (j+1)*N + k];
  temp += d_in[i*N*N + j*N + k-1];
  temp +=  d_in[i*N*N + j*N + k+1];
  d_out[i*N*N + j*N + k] = ((float) 0.8) *  temp;
  }

}

// TODO: Edit the function definition as required
__global__ void kernel2(float* d_in, float* d_out) {

  int i = (blockIdx.x*blockDim.x + threadIdx.x);
  int j = (blockIdx.y*blockDim.y + threadIdx.y);
  int k = (blockIdx.z*blockDim.z + threadIdx.z);

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int bx = blockDim.x-1;
  int by = blockDim.y-1;
  int bz = blockDim.z-1;


  __shared__ float tile3D[TileD][TileD][TileD];
  if(i<N && j<N && k<N){
    tile3D[tx][ty][tz] = d_in[i*N*N + j*N + k];
    __syncthreads();
  }

  if(i>0 && j>0 && k>0 && i<(N-1) && j<(N-1) && k<(N-1)){

    float a = (tx>0)?tile3D[tx-1][ty][tz]:d_in[(i-1)*N*N + j*N + k];
    float b = (tx<bx)?tile3D[tx+1][ty][tz]:d_in[(i+1)*N*N + j*N + k];

    float c = (ty>0)?tile3D[tx][ty-1][tz]:d_in[i*N*N + (j-1)*N + k];
    float d = (ty<by)?tile3D[tx][ty+1][tz]:d_in[i*N*N + (j+1)*N + k];

    float e = (tz>0)?tile3D[tx][ty][tz-1]:d_in[i*N*N + j*N + k-1];
    float f = (tz<bz)?tile3D[tx][ty][tz+1]:d_in[i*N*N + j*N + k+1];

    float temp = 0.0;

    temp = a+b;
    temp += c;
    temp += d;
    temp += e;
    temp += f;
    d_out[i*N*N + j*N + k] = ((float) 0.8)*temp;
  }

}

// TODO: Edit the function definition as required
__host__ void stencil(float* in, float* out) {

  for(int i=1;i<N-1;i++){
    for(int j=1;j<N-1;j++){
      for(int k=1;k<N-1;k++){
        out[i*N*N + j*N + k] = ((float) 0.8)*(in[(i-1)*N*N + j*N + k] + in[(i+1)*N*N + j*N + k] + in[i*N*N + (j-1)*N + k] + in[i*N*N + (j+1)*N + k] + in[i*N*N + j*N + k-1] +  in[i*N*N + j*N + k+1]);
      }
    }
  }

}

__host__ void check_result(float* w_ref, float* w_opt, uint64_t size) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        this_diff = w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

double rtclock() { // Seconds
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
  uint64_t SIZE = N * N * N;

  float *h_in = new float[SIZE];
  float *h_cpu_out = new float[SIZE];
  float *h_k1_out = new float[SIZE];
  float *h_k2_out = new float[SIZE];

  for(int i=0;i<SIZE;i++){
    h_in[i] = rand();
    h_cpu_out[i] = 0;
    h_k1_out[i] = 0;
    h_k2_out[i] = 0;
  }

  double clkbegin = rtclock();
  stencil(h_in,h_cpu_out);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;

  // TODO: Fill in kernel1

  float *d_k1_in, *d_k1_out;
  status = cudaMalloc(&d_k1_in, SIZE * sizeof(float));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }

  status = cudaMalloc(&d_k1_out, SIZE * sizeof(float));

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_k1_in, h_in, SIZE * sizeof(float), cudaMemcpyHostToDevice);
  int threadPerBlock1 = min(10,(int)N);
  int numBlock1 = (int)ceil(((double)N)/((double)threadPerBlock1));

  dim3 blockD1(threadPerBlock1,threadPerBlock1,threadPerBlock1);
  dim3 gridD1(numBlock1,numBlock1,numBlock1);

  kernel1<<<gridD1,blockD1>>>(d_k1_in,d_k1_out);

  cudaMemcpy(h_k1_out, d_k1_out, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  // TODO: Adapt check_result() and invoke
  float kernel_time;
  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  check_result(h_cpu_out,h_k1_out,N);
  std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";

  // TODO: Fill in kernel2
  float *d_k2_in, *d_k2_out;
  status = cudaMalloc(&d_k2_in, SIZE * sizeof(float));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }

  status = cudaMalloc(&d_k2_out, SIZE * sizeof(float));

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_k2_in, h_in, SIZE * sizeof(float), cudaMemcpyHostToDevice);
  int threadPerBlock2 = min(TileD,(int)N);
  int numBlock2 = (int)ceil(((double)N)/((double)threadPerBlock2));

  dim3 blockD2(threadPerBlock2,threadPerBlock2,threadPerBlock2);
  dim3 gridD2(numBlock2,numBlock2,numBlock2);

  kernel2<<<gridD2,blockD2>>>(d_k2_in,d_k2_out);

  cudaMemcpy(h_k2_out, d_k2_out, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  // TODO: Adapt check_result() and invoke
  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  check_result(h_cpu_out,h_k2_out,N);
  std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";

  // TODO: Free memory
  cudaFree(d_k1_in);
  cudaFree(d_k2_in);
  cudaFree(d_k1_out);
  cudaFree(d_k2_out);

  free(h_cpu_out);
  free(h_in);
  free(h_k1_out);
  free(h_k2_out);

  return EXIT_SUCCESS;
}
