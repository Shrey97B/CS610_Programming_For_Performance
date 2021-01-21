// Compile: nvcc -g -G -arch=sm_52 -std=c++11 assignment5-p4.cu -o assignment5-p4

#include <cmath>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

const uint64_t N = (1 << 10);
#define THRESHOLD (0.000001)

//Tile Dimensions should be less than or equal to 32 due to thread_per_block limitations and power of 2
#define TileDX (4)
#define TileDY (4)

using std::cerr;
using std::cout;
using std::endl;

__global__ void kernel1(uint64_t* d_A, uint64_t* d_B, uint64_t* d_C) {
  // TODO: Fill in

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int j = blockDim.y*blockIdx.y + threadIdx.y;

  if(i<N && j<N){
      uint64_t temp = 0;
      for(int k=0;k<N;k++){
          temp+=d_A[i*N + k]*d_B[k*N + j];
      }
      d_C[i*N + j] = temp;
  }

}

__global__ void kernel2(uint64_t* d_A, uint64_t* d_B, uint64_t* d_C) {
  // TODO: Fill in

   // TODO: Fill in
   int TileDimX=min(TileDX, min(32,(int)N));
   int TileDimY=min(TileDY, min(32,(int)N));
 
   int iv = blockDim.x*blockIdx.x + threadIdx.x;
   int jv = blockDim.y*blockIdx.y + threadIdx.y;
 
   int minn= min(TileDimX,TileDimY);
   __shared__ uint64_t tileA[TileDX][TileDY];
   __shared__ uint64_t tileB[TileDY][TileDY];
 
   int numtiles = (int)ceil(((double)N)/((double)minn));
   uint64_t temp = 0;
 
   for(int i=0;i<numtiles;i++){
       
       int rownumTileA = blockIdx.x;
       int colnumTileA = i;
       int rownumTileB = i;
       int colnumTileB = blockIdx.y;
 
       int r = threadIdx.x;
       int c = threadIdx.y;
       int indra = rownumTileA*TileDimX + r;
       int indca = colnumTileA*minn + c;
       int indrb = rownumTileB*minn + r;
       int indcb = colnumTileB*TileDimY + c;
  
       if(indra<N && indca<N && c<minn){
         tileA[r][c] = d_A[indra*N + indca];
       }
 
 
       if(indrb<N && indcb<N && r<minn){
         tileB[r][c] = d_B[indrb*N + indcb];
       }
 
       __syncthreads();
 
       int j=0;
       for(;j+3<minn;j+=4){
           temp+=tileA[r][j]*tileB[j][c];
           temp+=tileA[r][j+1]*tileB[j+1][c];
           temp+=tileA[r][j+2]*tileB[j+2][c];
           temp+=tileA[r][j+3]*tileB[j+3][c];
       }
       for(;j<minn;j++){
        temp+=tileA[r][j]*tileB[j][c];
       }
 
       __syncthreads();
 
   }
 
   if(iv<N && jv<N){
    d_C[iv*N + jv] = temp;
   }

}

__host__ void cpumatMul(uint64_t* h_A, uint64_t* h_B, uint64_t* h_C) {
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      float sum = 0.0;
      for (uint64_t k = 0; k < N; k++) {
        sum += h_A[i * N + k] * h_B[k * N + j];
      }
      h_C[i * N + j] = sum;
    }
  }
}

__host__ void check_result(uint64_t* w_ref, uint64_t* w_opt) {
  bool wrong = false;
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      if (w_ref[i * N + j] != w_opt[i * N + j]) {
        wrong = true;
        goto out;
      }
    }
  }
out:
  if (wrong) {
    cout << " Diffs found!" << endl;
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
  uint64_t SIZE = N * N;

  uint64_t *h_A, *h_B, *h_cpu_C, *h_gpu1_C, *h_gpu2_C;

  h_A = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_B = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_cpu_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_gpu1_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_gpu2_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      h_A[i * N + j] = rand() % 64;
      h_B[i * N + j] = 2;
      h_cpu_C[i * N + j] = 0;
      h_gpu1_C[i * N + j] = 0;
      h_gpu2_C[i * N + j] = 0;
    }
  }

  double clkbegin = rtclock();
  cpumatMul(h_A, h_B, h_cpu_C);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Matmul time on CPU: " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;

  uint64_t *d_A, *d_B, *d_C1;
  status = cudaMalloc(&d_A, SIZE * sizeof(uint64_t));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }

  status = cudaMalloc(&d_B, SIZE * sizeof(uint64_t));
  status = cudaMalloc(&d_C1, SIZE * sizeof(uint64_t));

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);
  status = cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);

  // TODO: Fill in

  int threadPerBlock1 = min(32,(int)N);
  int numBlock1 = (int)ceil(((double)N)/((double)threadPerBlock1));

  dim3 blockD1(threadPerBlock1,threadPerBlock1,1);
  dim3 gridD1(numBlock1,numBlock1,1);

  kernel1<<<gridD1,blockD1>>>(d_A,d_B,d_C1);

  cudaMemcpy(h_gpu1_C, d_C1, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  check_result(h_cpu_C, h_gpu1_C);
  float kernel_time;
  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";

  uint64_t* d_C2;
  status = cudaMalloc(&d_C2, SIZE * sizeof(uint64_t));
  // TODO: Fill in

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);
  status = cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);

  int threadPerBlock2x = min(TileDX, min(32,(int)N));
  int threadPerBlock2y = min(TileDY, min(32,(int)N));
  int numBlock2x = (int)ceil(((double)N)/((double)threadPerBlock2x));
  int numBlock2y = (int)ceil(((double)N)/((double)threadPerBlock2y));

  dim3 blockD2(threadPerBlock2x,threadPerBlock2y,1);
  dim3 gridD2(numBlock2x,numBlock2y,1);

  kernel2<<<gridD2,blockD2>>>(d_A,d_B,d_C2);

  cudaMemcpy(h_gpu2_C, d_C2, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  check_result(h_cpu_C, h_gpu2_C);
  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C1);
  cudaFree(d_C2);

  free(h_A);
  free(h_B);
  free(h_cpu_C);
  free(h_gpu1_C);
  free(h_gpu2_C);

  return EXIT_SUCCESS;
}
