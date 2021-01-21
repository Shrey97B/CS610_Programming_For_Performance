// Compile: nvcc -g -G -arch=sm_52 -std=c++11 assignment5-p3.cu -o assignment5-p3

#include <cmath>
#include <iostream>
#include <sys/time.h>

#define SIZE 4096
#define THRESHOLD (0.000001)

//Tile Dimension should be less than or equal to 32 due to thread_per_block limitations
#define TileDimen (1<<5)

using std::cerr;
using std::cout;
using std::endl;

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

__host__ void ATAonCPU(double** M, double** P) {
  for (int k = 0; k < SIZE; k++) {
    for (int i = 0; i < SIZE; i++) {
      for (int j = 0; j < SIZE; j++)
        P[i][j] += M[k][i] * M[k][j];
    }
  }
}

__host__ void check_result(double** Test, double** Ref) {
  double maxdiff = 0, rel_diff = 0;
  int numdiffs = 0;

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      rel_diff = (Test[i][j] - Ref[i][j]);
      if (fabs(rel_diff) > THRESHOLD) {
        numdiffs++;
        if (rel_diff > maxdiff)
          maxdiff = rel_diff;
      }
    }
  }
  if (numdiffs > 0)
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << " Max Diff = " << maxdiff
         << "\n";
  else
    cout << "No differences found between base and test versions\n";
}

__global__ void ATAkernel(double* A, double* B) {
  // TODO: Fill in

  int TileDim=0;
  if(TileDimen>=SIZE){
      TileDim = SIZE;
  }
  else{
      TileDim = TileDimen;
  }
  int iv = blockDim.x*blockIdx.x + threadIdx.x;
  int jv = blockDim.y*blockIdx.y + threadIdx.y;

  __shared__ double tileA[TileDimen][TileDimen];
  __shared__ double tileAT[TileDimen][TileDimen];

  int numtiles = SIZE/TileDim;
  double temp = 0;

  for(int i=0;i<numtiles;i++){
      
      int rowTileA = blockIdx.x;
      int colTileA = i;
      int rowTileAT = i;
      int colTileAT = blockIdx.y;

      int r = threadIdx.x, c = threadIdx.y;
      tileA[r][c] = A[(rowTileA*TileDim + r)*SIZE + colTileA*TileDim + c];
      tileAT[r][c] = A[(colTileAT*TileDim + c)*SIZE + rowTileAT*TileDim + r];
      __syncthreads();

      int x = threadIdx.x;
      int y = threadIdx.y;

      if(iv<=jv){
        //unrolling below loop
        int j=0;
        for(;j+3<TileDim;j+=4){
          temp+=tileA[x][j]*tileAT[j][y];
          temp+= tileA[x][j+1]*tileAT[j+1][y];
          temp+= tileA[x][j+2]*tileAT[j+2][y];
          temp+= tileA[x][j+3]*tileAT[j+3][y];
        }
        for(;j<TileDimen;j++){
          temp+=tileA[x][j]*tileAT[j][y];
        }
      }

      __syncthreads();

  }

  if(iv<SIZE && jv<SIZE && iv<=jv){
    B[iv*SIZE + jv] = temp;
    B[jv*SIZE + iv] = temp;
  }

}

__global__ void ATAkernel2(double* A, double* B) {
  // TODO: Fill in

  int i = (blockDim.x*blockIdx.x + threadIdx.x);
  int j = (blockDim.y*blockIdx.y + threadIdx.y);

  if(i<SIZE && j<SIZE && i<=j){
    for(int k=0;k<SIZE;k++){
      B[i*SIZE + j] += A[k*SIZE + i]*A[k*SIZE + j];
    }
    B[j*SIZE + i] = B[i*SIZE + j];
  }

}

int main() {
  cout << "Matrix Size = " << SIZE << "\n";

  double** h_in = new double*[SIZE];
  for (int i = 0; i < SIZE; i++) {
    h_in[i] = new double[SIZE];
  }

  double** h_cpu_out = new double*[SIZE];
  for (int i = 0; i < SIZE; i++) {
    h_cpu_out[i] = new double[SIZE];
  }

  double** h_dev_out = new double*[SIZE];
  double** h_dev_out_2 = new double*[SIZE];
  for (int i = 0; i < SIZE; i++) {
    h_dev_out[i] = new double[SIZE];
    h_dev_out_2[i] = new double[SIZE];
  }

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      h_in[i][j] = i * j * 0.25;
      h_cpu_out[i][j] = 0;
      h_dev_out[i][j] = 0;
      h_dev_out_2[i][j] = 0;
    }
  }

  double clkbegin = rtclock();
  ATAonCPU(h_in, h_cpu_out);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "A^T.A on CPU: " << ((2.0 * SIZE * SIZE * SIZE) / cpu_time)
       << " GFLOPS; Time = " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;
  double* d_in;
  double* d_out;
  float kernel_time;
  // TODO: Fill in

  int matsizes = SIZE*SIZE*sizeof(double);

  status = cudaMalloc(&d_in,matsizes);
  status = cudaMalloc(&d_out,matsizes);
  if(status!=cudaSuccess){
    cout<<"Cuda Malloc Failed"<<endl;
  }

  double* tempin = new double[SIZE*SIZE];
  for(int i=0;i<SIZE*SIZE;i++){
    tempin[i] = h_in[(i/SIZE)][i%SIZE];
  }

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, 0);

  status = cudaMemcpy(d_in, tempin, matsizes, cudaMemcpyHostToDevice);


  int TileDim = min(TileDimen,min((1<<5),SIZE));

  int threadPB = TileDim;
  int numPB = (int)ceil(((double)(SIZE))/((double)threadPB));

  dim3 GridD(numPB,numPB,1);
  dim3 BlockD(threadPB,threadPB,1);

  ATAkernel<<<GridD,BlockD>>>(d_in,d_out);

  double* tempout = new double[SIZE*SIZE];
  status = cudaMemcpy(tempout, d_out, matsizes, cudaMemcpyDeviceToHost);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  for(int i=0;i<SIZE*SIZE;i++){
    h_dev_out[(i/SIZE)][i%SIZE] = tempout[i];
  }

  cudaEventElapsedTime(&kernel_time, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cout << "A^T.A on GPU: " << ((2.0 * SIZE * SIZE * SIZE) / (kernel_time * 1.0e-03))
       << " GFLOPS; Time = " << kernel_time << " msec" << endl;

  check_result(h_cpu_out, h_dev_out);

  cudaError_t status2;
  cudaEvent_t start2, end2;
  double* d_in_2;
  double* d_out_2;
  float kernel_time_2;
  // TODO: Fill in

  status2 = cudaMalloc(&d_in_2,matsizes);
  status2 = cudaMalloc(&d_out_2,matsizes);
  if(status2!=cudaSuccess){
    cout<<"Cuda Malloc Failed"<<endl;
  }

  double* tempin_2 = new double[SIZE*SIZE];
  for(int i=0;i<SIZE*SIZE;i++){
    tempin_2[i] = h_in[(i/SIZE)][i%SIZE];
  }

  
  cudaEventCreate(&start2);
  cudaEventCreate(&end2);

  cudaEventRecord(start2, 0);

  status2 = cudaMemcpy(d_in_2, tempin_2, matsizes, cudaMemcpyHostToDevice);


  int threadPB2 = min(32,SIZE);
  int numPB2 = (int)ceil(((double)(SIZE))/((double)threadPB2));

  dim3 GridD2(numPB2,numPB2,1);
  dim3 BlockD2(threadPB2,threadPB2,1);

  ATAkernel2<<<GridD2,BlockD2>>>(d_in_2,d_out_2);

  double* tempout_2 = new double[SIZE*SIZE];
  status2 = cudaMemcpy(tempout_2, d_out_2, matsizes, cudaMemcpyDeviceToHost);

  cudaEventRecord(end2, 0);
  cudaEventSynchronize(end2);

  for(int i=0;i<SIZE*SIZE;i++){
    h_dev_out_2[(i/SIZE)][i%SIZE] = tempout_2[i];
  }

  cudaEventElapsedTime(&kernel_time_2, start2, end2);
  cudaEventDestroy(start2);
  cudaEventDestroy(end2);
  cout << "A^T.A 2 on GPU: " << ((2.0 * SIZE * SIZE * SIZE) / (kernel_time * 1.0e-03))
       << " GFLOPS; Time = " << kernel_time_2 << " msec" << endl;

  check_result(h_cpu_out, h_dev_out_2);

  // Free device memory
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_in_2);
  cudaFree(d_out_2);
  
  free(h_in);
  free(h_cpu_out);
  free(h_dev_out);
  free(h_dev_out_2);

  return EXIT_SUCCESS;
}
