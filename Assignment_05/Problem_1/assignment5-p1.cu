// Compile: nvcc -g -G -arch=sm_52 -std=c++11 assignment5-p1.cu -o assignment5-p1

#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <iostream>
#include <new>
#include <sys/time.h>

#define THRESHOLD (0.000001)

#define SIZE1 8192
#define SIZE2 (SIZE1+8)
#define ITER 100

using std::cerr;
using std::cout;
using std::endl;

__global__ void kernel1(double* d_k1_in) {
  // TODO: Fill in
  int indexcomb = blockDim.x * blockIdx.x + threadIdx.x;
  int j = indexcomb;

  if(j<(SIZE1-1)){
    for(int k=0;k<ITER;k++){
      for(int i=1;i<(SIZE1-1);i++){
        double temp=0.0;
        temp = d_k1_in[(i - 1)*SIZE1 + j + 1] + d_k1_in[i*SIZE1 + j+1];
        temp += d_k1_in[(i+1)*SIZE1 + j+1];
        d_k1_in[i*SIZE1 + j+1] = temp;
      }
    }
  }
}

__global__ void kernel2(double* d_k2_in) {
  // TODO: Fill in
  int indexcomb = blockDim.x * blockIdx.x + threadIdx.x;
  int j = indexcomb;

  if(j<(SIZE2-1)){
    for(int k=0;k<ITER;k++){
      int i=1;
      //loop unrolled over 4 ways
      for(;i+3<(SIZE2-1);i+=4){
        double temp=d_k2_in[(i - 1)*SIZE2 + j + 1] + d_k2_in[i*SIZE2 + j+1];
        temp += d_k2_in[(i+1)*SIZE2 + j+1];
        d_k2_in[i*SIZE2 + j+1] = temp;

        temp=d_k2_in[(i)*SIZE2 + j + 1] + d_k2_in[(i+1)*SIZE2 + j+1];
        temp += d_k2_in[(i+2)*SIZE2 + j+1];
        d_k2_in[(i+1)*SIZE2 + j+1] = temp;

        temp=d_k2_in[(i+1)*SIZE2 + j + 1] + d_k2_in[(i+2)*SIZE2 + j+1];
        temp += d_k2_in[(i+3)*SIZE2 + j+1];
        d_k2_in[(i+2)*SIZE2 + j+1] = temp;

        temp=d_k2_in[(i+2)*SIZE2 + j + 1] + d_k2_in[(i+3)*SIZE2 + j+1];
        temp += d_k2_in[(i+4)*SIZE2 + j+1];
        d_k2_in[(i+3)*SIZE2 + j+1] = temp;

      }

      for(;i<(SIZE2-1);i++){
        double temp=d_k2_in[(i - 1)*SIZE2 + j + 1] + d_k2_in[i*SIZE2 + j+1];
        temp += d_k2_in[(i+1)*SIZE2 + j+1];
        d_k2_in[i*SIZE2 + j+1] = temp;
      }

    }
  }
}

__host__ void serial(double** h_ser_in) {
  for (int k = 0; k < ITER; k++) {
    for (int i = 1; i < (SIZE1 - 1); i++) {
      for (int j = 0; j < (SIZE1 - 1); j++) {
        h_ser_in[i][j + 1] =
            (h_ser_in[i - 1][j + 1] + h_ser_in[i][j + 1] + h_ser_in[i + 1][j + 1]);
      }
    }
  }
}

__host__ void check_result(double** w_ref, double** w_opt, uint64_t size) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      this_diff = w_ref[i][j] - w_opt[i][j];
      if (fabs(this_diff) > THRESHOLD) {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
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
  double** h_ser_in = new double*[SIZE1];
  double** h_ser_out = new double*[SIZE1];
    
  double** h_k1_in = new double*[SIZE1];
  double** h_k1_out = new double*[SIZE1];

  for (int i = 0; i < SIZE1; i++) {
    h_ser_in[i] = new double[SIZE1];
    h_ser_out[i] = new double[SIZE1];
    h_k1_in[i] = new double[SIZE1];
    h_k1_out[i] = new double[SIZE1];
  }

  for (int i = 0; i < SIZE1; i++) {
    for (int j = 0; j < SIZE1; j++) {
      h_ser_in[i][j] = 1;
      h_ser_out[i][j] = 0;
      h_k1_in[i][j] = 1;
      h_k1_out[i][j] = 0;
    }
  }

  double** h_k2_in = new double*[SIZE2];
  double** h_k2_out = new double*[SIZE2];
  for (int i = 0; i < SIZE2; i++) {
    h_k2_in[i] = new double[SIZE2];
    h_k2_out[i] = new double[SIZE2];
  }

  for (int i = 0; i < SIZE2; i++) {
    for (int j = 0; j < SIZE2; j++) {
      h_k2_in[i][j] = 1;
      h_k2_out[i][j] = 0;
    }
  }

  double clkbegin = rtclock();
  serial(h_ser_in);
  double clkend = rtclock();
  double time = clkend - clkbegin; // seconds
  cout << "Serial code on CPU: " << ((2.0 * SIZE1 * SIZE1 * ITER) / time)
       << " GFLOPS; Time = " << time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;
  float k1_time; // ms

  double* d_k1_in;
  double* d_k1_out;
  // TODO: Fill in

  status = cudaMalloc(&d_k1_in,SIZE1*SIZE1*sizeof(double));
  if(status!=cudaSuccess){
    cout<<"Error in cuda malloc"<<endl;
  }

  double *tempinarr1 = new double[SIZE1*SIZE1];
  for(int i=0;i<SIZE1*SIZE1;i++){
     tempinarr1[i] = h_k1_in[(i/SIZE1)][(i%SIZE1)];
  }

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_k1_in, tempinarr1, SIZE1*SIZE1*sizeof(double), cudaMemcpyHostToDevice);

  int threadPerBlock1 = 1024;
  int numBlock1 = (int) ceil(((double)SIZE1)/((double)threadPerBlock1));

  kernel1<<<numBlock1,threadPerBlock1>>>(d_k1_in);

  status = cudaMemcpy(tempinarr1, d_k1_in, SIZE1*SIZE1*sizeof(double), cudaMemcpyDeviceToHost);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&k1_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  for(int i=0;i<SIZE1*SIZE1;i++){
    h_k1_in[(i/SIZE1)][(i%SIZE1)] = tempinarr1[i];
  }

  check_result(h_ser_in, h_k1_in, SIZE1);
  cout << "Kernel 1 on GPU: " << ((2.0 * SIZE1 * SIZE1 * ITER) / (k1_time * 1.0e-3))
       << " GFLOPS; Time = " << k1_time << " msec" << endl;

  double* d_k2_in;
  double* d_k2_out;
  // TODO: Fill in

  status = cudaMalloc(&d_k2_in,SIZE2*SIZE2*sizeof(double));
  if(status!=cudaSuccess){
    cout<<"Error in cuda malloc"<<endl;
  }

  double *tempinarr2 = new double[SIZE2*SIZE2];
  for(int i=0;i<SIZE2*SIZE2;i++){
     tempinarr2[i] = h_k2_in[(i/SIZE2)][(i%SIZE2)];
  }

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_k2_in, tempinarr2, SIZE2*SIZE2*sizeof(double), cudaMemcpyHostToDevice);

  int threadPerBlock2 = min(SIZE2,1024);
  int numBlock2 = (int) ceil(((double)SIZE2)/((double)threadPerBlock2));

  kernel2<<<numBlock2,threadPerBlock2>>>(d_k2_in);

  status = cudaMemcpy(tempinarr2, d_k2_in, SIZE2*SIZE2*sizeof(double), cudaMemcpyDeviceToHost);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&k1_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  for(int i=0;i<SIZE2*SIZE2;i++){
    h_k2_in[(i/SIZE2)][(i%SIZE2)] = tempinarr2[i];
  }

  cout << "Kernel 2 on GPU: " << ((2.0 * SIZE2 * SIZE2 * ITER) / (k1_time * 1.0e-3))
       << " GFLOPS; Time = " << k1_time << " msec" << endl;

  cudaFree(d_k1_in);
  cudaFree(d_k2_in);

  for (int i = 0; i < SIZE1; i++) {
    delete[] h_ser_in[i];
    delete[] h_ser_out[i];
    delete[] h_k1_in[i];
    delete[] h_k1_out[i];
  }
  delete[] h_ser_in;
  delete[] h_ser_out;
  delete[] h_k1_in;
  delete[] h_k1_out;

  for (int i = 0; i < SIZE2; i++) {
    delete[] h_k2_in[i];
    delete[] h_k2_out[i];
  }
  delete[] h_k2_in;
  delete[] h_k2_out;

  return EXIT_SUCCESS;
}
