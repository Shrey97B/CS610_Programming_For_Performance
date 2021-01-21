/**
 * g++ -o problem5 problem5.cpp -lpthread
 * ./problem5
 */

// TODO: This file is just a template, feel free to modify it to suit your needs

#include <cstring>
#include <iostream>
#include <pthread.h>
#include <sys/time.h>

using std::cout;
using std::endl;

const uint16_t NUM_THREADS = 2; // TODO: You may want to change this
const uint16_t MAT_SIZE = 4096;

void sequential_matmul();
void parallel_matmul();
// TODO: Other function definitions
void sequential_matmul_opt();
void parallel_matmul_opt();

double rtclock();
void check_result(uint64_t*, uint64_t*);
const double THRESHOLD = 0.0000001;

uint64_t* matrix_A;
uint64_t* matrix_B;
uint64_t* sequential_C;
uint64_t* sequential_opt_C;
uint64_t* parallel_C;
uint64_t* parallel_opt_C;

uint16_t block_size;
uint16_t unroll_size = 8;

void* parallel_work(void* thread_id) { 
  
  int *thid = (int *) thread_id;
  int numd = MAT_SIZE/NUM_THREADS;
  int ibeg = (*thid)*numd;
  int iupb = (*thid+1)*(numd);
   int i, j, k;
  for (i = ibeg; i < iupb; i++) {
    for (j = 0; j < MAT_SIZE; j++) {
      uint16_t temp = 0;
      for (k = 0; k < MAT_SIZE; k++)
        temp += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
        parallel_C[i * MAT_SIZE + j] = temp;
    }
  }
  return nullptr; 
  
}

void sequential_matmul() {
  int i, j, k;
  for (i = 0; i < MAT_SIZE; i++) {
    for (j = 0; j < MAT_SIZE; j++) {
      uint16_t temp = 0;
      for (k = 0; k < MAT_SIZE; k++)
        temp += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
      sequential_C[i * MAT_SIZE + j] = temp;
    }
  }
}

void sequential_matmul_opt() {
  // TODO:
  int i, j, k, i1, j1, k1;
  for(i1=0;i1<MAT_SIZE;i1+=block_size){
    for(j1=0;j1<MAT_SIZE;j1+=block_size){
      for(k1=0;k1<MAT_SIZE;k1+=block_size){
        for (i = i1; i < (i1 + block_size); i++) {
           for (j = j1; j < (j1 + block_size); j+=unroll_size) {
               uint16_t temp[unroll_size] = {0};
               temp[0] = sequential_opt_C[i * MAT_SIZE + j];
               temp[1] = sequential_opt_C[i * MAT_SIZE + j+1];
               temp[2] = sequential_opt_C[i * MAT_SIZE + j+2];
               temp[3] = sequential_opt_C[i * MAT_SIZE + j+3];
               temp[4] = sequential_opt_C[i * MAT_SIZE + j+4];
               temp[5] = sequential_opt_C[i * MAT_SIZE + j+5];
               temp[6] = sequential_opt_C[i * MAT_SIZE + j+6];
               temp[7] = sequential_opt_C[i * MAT_SIZE + j+7];
              for (k = k1; k < (k1 + block_size); k+=unroll_size){

		temp[0] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]) + (matrix_A[i * MAT_SIZE + k+1] * matrix_B[(k+1) * MAT_SIZE + j]) + (matrix_A[i * MAT_SIZE + k+2] * matrix_B[(k+2) * MAT_SIZE + j]) + (matrix_A[i * MAT_SIZE + (k+3)] * matrix_B[(k+3) * MAT_SIZE + j]) + (matrix_A[i * MAT_SIZE + k+4] * matrix_B[(k+4) * MAT_SIZE + j]) + (matrix_A[i * MAT_SIZE + k+5] * matrix_B[(k+5) * MAT_SIZE + j]) + (matrix_A[i * MAT_SIZE + k+6] * matrix_B[(k+6) * MAT_SIZE + j]) + (matrix_A[i * MAT_SIZE + (k+7)] * matrix_B[(k+7) * MAT_SIZE + j]);

		temp[1] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]) + (matrix_A[i * MAT_SIZE + k+1] * matrix_B[(k+1) * MAT_SIZE + j+1]) + (matrix_A[i * MAT_SIZE + k+2] * matrix_B[(k+2) * MAT_SIZE + j+1]) + (matrix_A[i * MAT_SIZE + (k+3)] * matrix_B[(k+3) * MAT_SIZE + j+1]) + (matrix_A[i * MAT_SIZE + k+4] * matrix_B[(k+4) * MAT_SIZE + j+1]) + (matrix_A[i * MAT_SIZE + k+5] * matrix_B[(k+5) * MAT_SIZE + j+1]) + (matrix_A[i * MAT_SIZE + k+6] * matrix_B[(k+6) * MAT_SIZE + j+1]) + (matrix_A[i * MAT_SIZE + (k+7)] * matrix_B[(k+7) * MAT_SIZE + j+1]);

		temp[2] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]) + (matrix_A[i * MAT_SIZE + k+1] * matrix_B[(k+1) * MAT_SIZE + j+2]) + (matrix_A[i * MAT_SIZE + k+2] * matrix_B[(k+2) * MAT_SIZE + j+2]) + (matrix_A[i * MAT_SIZE + (k+3)] * matrix_B[(k+3) * MAT_SIZE + j+2]) + (matrix_A[i * MAT_SIZE + k+4] * matrix_B[(k+4) * MAT_SIZE + j+2]) + (matrix_A[i * MAT_SIZE + k+5] * matrix_B[(k+5) * MAT_SIZE + j+2]) + (matrix_A[i * MAT_SIZE + k+6] * matrix_B[(k+6) * MAT_SIZE + j+2]) + (matrix_A[i * MAT_SIZE + (k+7)] * matrix_B[(k+7) * MAT_SIZE + j+2]);

		temp[3] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]) + (matrix_A[i * MAT_SIZE + k+1] * matrix_B[(k+1) * MAT_SIZE + j+3]) + (matrix_A[i * MAT_SIZE + k+2] * matrix_B[(k+2) * MAT_SIZE + j+3]) + (matrix_A[i * MAT_SIZE + (k+3)] * matrix_B[(k+3) * MAT_SIZE + j+3]) + (matrix_A[i * MAT_SIZE + k+4] * matrix_B[(k+4) * MAT_SIZE + j+3]) + (matrix_A[i * MAT_SIZE + k+5] * matrix_B[(k+5) * MAT_SIZE + j+3]) + (matrix_A[i * MAT_SIZE + k+6] * matrix_B[(k+6) * MAT_SIZE + j+3]) + (matrix_A[i * MAT_SIZE + (k+7)] * matrix_B[(k+7) * MAT_SIZE + j+3]);

		temp[4] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]) + (matrix_A[i * MAT_SIZE + k+1] * matrix_B[(k+1) * MAT_SIZE + j+4]) + (matrix_A[i * MAT_SIZE + k+2] * matrix_B[(k+2) * MAT_SIZE + j+4]) + (matrix_A[i * MAT_SIZE + (k+3)] * matrix_B[(k+3) * MAT_SIZE + j+4]) + (matrix_A[i * MAT_SIZE + k+4] * matrix_B[(k+4) * MAT_SIZE + j+4]) + (matrix_A[i * MAT_SIZE + k+5] * matrix_B[(k+5) * MAT_SIZE + j+4]) + (matrix_A[i * MAT_SIZE + k+6] * matrix_B[(k+6) * MAT_SIZE + j+4]) + (matrix_A[i * MAT_SIZE + (k+7)] * matrix_B[(k+7) * MAT_SIZE + j+4]);

		temp[5] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]) + (matrix_A[i * MAT_SIZE + k+1] * matrix_B[(k+1) * MAT_SIZE + j+5]) + (matrix_A[i * MAT_SIZE + k+2] * matrix_B[(k+2) * MAT_SIZE + j+5]) + (matrix_A[i * MAT_SIZE + (k+3)] * matrix_B[(k+3) * MAT_SIZE + j+5]) + (matrix_A[i * MAT_SIZE + k+4] * matrix_B[(k+4) * MAT_SIZE + j+5]) + (matrix_A[i * MAT_SIZE + k+5] * matrix_B[(k+5) * MAT_SIZE + j+5]) + (matrix_A[i * MAT_SIZE + k+6] * matrix_B[(k+6) * MAT_SIZE + j+5]) + (matrix_A[i * MAT_SIZE + (k+7)] * matrix_B[(k+7) * MAT_SIZE + j+5]);

		temp[6] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]) + (matrix_A[i * MAT_SIZE + k+1] * matrix_B[(k+1) * MAT_SIZE + j+6]) + (matrix_A[i * MAT_SIZE + k+2] * matrix_B[(k+2) * MAT_SIZE + j+6]) + (matrix_A[i * MAT_SIZE + (k+3)] * matrix_B[(k+3) * MAT_SIZE + j+6]) + (matrix_A[i * MAT_SIZE + k+4] * matrix_B[(k+4) * MAT_SIZE + j+6]) + (matrix_A[i * MAT_SIZE + k+5] * matrix_B[(k+5) * MAT_SIZE + j+6]) + (matrix_A[i * MAT_SIZE + k+6] * matrix_B[(k+6) * MAT_SIZE + j+6]) + (matrix_A[i * MAT_SIZE + (k+7)] * matrix_B[(k+7) * MAT_SIZE + j+6]);

		temp[7] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]) + (matrix_A[i * MAT_SIZE + k+1] * matrix_B[(k+1) * MAT_SIZE + j+7]) + (matrix_A[i * MAT_SIZE + k+2] * matrix_B[(k+2) * MAT_SIZE + j+7]) + (matrix_A[i * MAT_SIZE + (k+3)] * matrix_B[(k+3) * MAT_SIZE + j+7]) + (matrix_A[i * MAT_SIZE + k+4] * matrix_B[(k+4) * MAT_SIZE + j+7]) + (matrix_A[i * MAT_SIZE + k+5] * matrix_B[(k+5) * MAT_SIZE + j+7]) + (matrix_A[i * MAT_SIZE + k+6] * matrix_B[(k+6) * MAT_SIZE + j+7]) + (matrix_A[i * MAT_SIZE + (k+7)] * matrix_B[(k+7) * MAT_SIZE + j+7]);
		
		}
		sequential_opt_C[i * MAT_SIZE + j] = temp[0];
		sequential_opt_C[i * MAT_SIZE + j+1] = temp[1];
		sequential_opt_C[i * MAT_SIZE + j+2] = temp[2];
		sequential_opt_C[i * MAT_SIZE + j+3] = temp[3];
		sequential_opt_C[i * MAT_SIZE + j+4] = temp[4];
		sequential_opt_C[i * MAT_SIZE + j+5] = temp[5];
		sequential_opt_C[i * MAT_SIZE + j+6] = temp[6];
		sequential_opt_C[i * MAT_SIZE + j+7] = temp[7];
	  }
      }
     }
    }
  }

}

void* parallel_matmul_opt(void* thread_id) {
  // TODO:

  int *thid = (int *) thread_id;
  int numd = MAT_SIZE/NUM_THREADS;
  int ibeg = (*thid)*numd;
  int iupb = (*thid+1)*(numd);
  int i, j, k, i1, j1, k1;
  for(i1=ibeg;i1<iupb;i1+=block_size){
    for(j1=0;j1<MAT_SIZE;j1+=block_size){
      for(k1=0;k1<MAT_SIZE;k1+=block_size){
        for (i = i1; i < (i1 + block_size); i++) {
           for (j = j1; j < (j1 + block_size); j+=unroll_size) {
               uint16_t temp[unroll_size] = {0};
               temp[0] = parallel_opt_C[i * MAT_SIZE + j];
               temp[1] = parallel_opt_C[i * MAT_SIZE + j+1];
               temp[2] = parallel_opt_C[i * MAT_SIZE + j+2];
               temp[3] = parallel_opt_C[i * MAT_SIZE + j+3];
               temp[4] = parallel_opt_C[i * MAT_SIZE + j+4];
               temp[5] = parallel_opt_C[i * MAT_SIZE + j+5];
               temp[6] = parallel_opt_C[i * MAT_SIZE + j+6];
               temp[7] = parallel_opt_C[i * MAT_SIZE + j+7];
              for (k = k1; k < (k1 + block_size); k+=unroll_size){

		temp[0] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]) + (matrix_A[i * MAT_SIZE + k+1] * matrix_B[(k+1) * MAT_SIZE + j]) + (matrix_A[i * MAT_SIZE + k+2] * matrix_B[(k+2) * MAT_SIZE + j]) + (matrix_A[i * MAT_SIZE + (k+3)] * matrix_B[(k+3) * MAT_SIZE + j]) + (matrix_A[i * MAT_SIZE + k+4] * matrix_B[(k+4) * MAT_SIZE + j]) + (matrix_A[i * MAT_SIZE + k+5] * matrix_B[(k+5) * MAT_SIZE + j]) + (matrix_A[i * MAT_SIZE + k+6] * matrix_B[(k+6) * MAT_SIZE + j]) + (matrix_A[i * MAT_SIZE + (k+7)] * matrix_B[(k+7) * MAT_SIZE + j]);

		temp[1] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+1]) + (matrix_A[i * MAT_SIZE + k+1] * matrix_B[(k+1) * MAT_SIZE + j+1]) + (matrix_A[i * MAT_SIZE + k+2] * matrix_B[(k+2) * MAT_SIZE + j+1]) + (matrix_A[i * MAT_SIZE + (k+3)] * matrix_B[(k+3) * MAT_SIZE + j+1]) + (matrix_A[i * MAT_SIZE + k+4] * matrix_B[(k+4) * MAT_SIZE + j+1]) + (matrix_A[i * MAT_SIZE + k+5] * matrix_B[(k+5) * MAT_SIZE + j+1]) + (matrix_A[i * MAT_SIZE + k+6] * matrix_B[(k+6) * MAT_SIZE + j+1]) + (matrix_A[i * MAT_SIZE + (k+7)] * matrix_B[(k+7) * MAT_SIZE + j+1]);

		temp[2] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+2]) + (matrix_A[i * MAT_SIZE + k+1] * matrix_B[(k+1) * MAT_SIZE + j+2]) + (matrix_A[i * MAT_SIZE + k+2] * matrix_B[(k+2) * MAT_SIZE + j+2]) + (matrix_A[i * MAT_SIZE + (k+3)] * matrix_B[(k+3) * MAT_SIZE + j+2]) + (matrix_A[i * MAT_SIZE + k+4] * matrix_B[(k+4) * MAT_SIZE + j+2]) + (matrix_A[i * MAT_SIZE + k+5] * matrix_B[(k+5) * MAT_SIZE + j+2]) + (matrix_A[i * MAT_SIZE + k+6] * matrix_B[(k+6) * MAT_SIZE + j+2]) + (matrix_A[i * MAT_SIZE + (k+7)] * matrix_B[(k+7) * MAT_SIZE + j+2]);

		temp[3] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+3]) + (matrix_A[i * MAT_SIZE + k+1] * matrix_B[(k+1) * MAT_SIZE + j+3]) + (matrix_A[i * MAT_SIZE + k+2] * matrix_B[(k+2) * MAT_SIZE + j+3]) + (matrix_A[i * MAT_SIZE + (k+3)] * matrix_B[(k+3) * MAT_SIZE + j+3]) + (matrix_A[i * MAT_SIZE + k+4] * matrix_B[(k+4) * MAT_SIZE + j+3]) + (matrix_A[i * MAT_SIZE + k+5] * matrix_B[(k+5) * MAT_SIZE + j+3]) + (matrix_A[i * MAT_SIZE + k+6] * matrix_B[(k+6) * MAT_SIZE + j+3]) + (matrix_A[i * MAT_SIZE + (k+7)] * matrix_B[(k+7) * MAT_SIZE + j+3]);

		temp[4] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+4]) + (matrix_A[i * MAT_SIZE + k+1] * matrix_B[(k+1) * MAT_SIZE + j+4]) + (matrix_A[i * MAT_SIZE + k+2] * matrix_B[(k+2) * MAT_SIZE + j+4]) + (matrix_A[i * MAT_SIZE + (k+3)] * matrix_B[(k+3) * MAT_SIZE + j+4]) + (matrix_A[i * MAT_SIZE + k+4] * matrix_B[(k+4) * MAT_SIZE + j+4]) + (matrix_A[i * MAT_SIZE + k+5] * matrix_B[(k+5) * MAT_SIZE + j+4]) + (matrix_A[i * MAT_SIZE + k+6] * matrix_B[(k+6) * MAT_SIZE + j+4]) + (matrix_A[i * MAT_SIZE + (k+7)] * matrix_B[(k+7) * MAT_SIZE + j+4]);

		temp[5] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+5]) + (matrix_A[i * MAT_SIZE + k+1] * matrix_B[(k+1) * MAT_SIZE + j+5]) + (matrix_A[i * MAT_SIZE + k+2] * matrix_B[(k+2) * MAT_SIZE + j+5]) + (matrix_A[i * MAT_SIZE + (k+3)] * matrix_B[(k+3) * MAT_SIZE + j+5]) + (matrix_A[i * MAT_SIZE + k+4] * matrix_B[(k+4) * MAT_SIZE + j+5]) + (matrix_A[i * MAT_SIZE + k+5] * matrix_B[(k+5) * MAT_SIZE + j+5]) + (matrix_A[i * MAT_SIZE + k+6] * matrix_B[(k+6) * MAT_SIZE + j+5]) + (matrix_A[i * MAT_SIZE + (k+7)] * matrix_B[(k+7) * MAT_SIZE + j+5]);

		temp[6] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+6]) + (matrix_A[i * MAT_SIZE + k+1] * matrix_B[(k+1) * MAT_SIZE + j+6]) + (matrix_A[i * MAT_SIZE + k+2] * matrix_B[(k+2) * MAT_SIZE + j+6]) + (matrix_A[i * MAT_SIZE + (k+3)] * matrix_B[(k+3) * MAT_SIZE + j+6]) + (matrix_A[i * MAT_SIZE + k+4] * matrix_B[(k+4) * MAT_SIZE + j+6]) + (matrix_A[i * MAT_SIZE + k+5] * matrix_B[(k+5) * MAT_SIZE + j+6]) + (matrix_A[i * MAT_SIZE + k+6] * matrix_B[(k+6) * MAT_SIZE + j+6]) + (matrix_A[i * MAT_SIZE + (k+7)] * matrix_B[(k+7) * MAT_SIZE + j+6]);

		temp[7] += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j+7]) + (matrix_A[i * MAT_SIZE + k+1] * matrix_B[(k+1) * MAT_SIZE + j+7]) + (matrix_A[i * MAT_SIZE + k+2] * matrix_B[(k+2) * MAT_SIZE + j+7]) + (matrix_A[i * MAT_SIZE + (k+3)] * matrix_B[(k+3) * MAT_SIZE + j+7]) + (matrix_A[i * MAT_SIZE + k+4] * matrix_B[(k+4) * MAT_SIZE + j+7]) + (matrix_A[i * MAT_SIZE + k+5] * matrix_B[(k+5) * MAT_SIZE + j+7]) + (matrix_A[i * MAT_SIZE + k+6] * matrix_B[(k+6) * MAT_SIZE + j+7]) + (matrix_A[i * MAT_SIZE + (k+7)] * matrix_B[(k+7) * MAT_SIZE + j+7]);
		
		}
		parallel_opt_C[i * MAT_SIZE + j] = temp[0];
		parallel_opt_C[i * MAT_SIZE + j+1] = temp[1];
		parallel_opt_C[i * MAT_SIZE + j+2] = temp[2];
		parallel_opt_C[i * MAT_SIZE + j+3] = temp[3];
		parallel_opt_C[i * MAT_SIZE + j+4] = temp[4];
		parallel_opt_C[i * MAT_SIZE + j+5] = temp[5];
		parallel_opt_C[i * MAT_SIZE + j+6] = temp[6];
		parallel_opt_C[i * MAT_SIZE + j+7] = temp[7];
	  }
      }
     }
    }
  }

}

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    printf("Error return from gettimeofday: %d\n", stat);
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void check_result(uint64_t* first_res, uint64_t* second_res) {
  double maxdiff, this_diff;
  int numdiffs;
  int i, j;
  numdiffs = 0;
  maxdiff = 0;

  for (i = 0; i < MAT_SIZE; i++) {
    for (j = 0; j < MAT_SIZE; j++) {
      this_diff = first_res[i * MAT_SIZE + j] - second_res[i * MAT_SIZE + j];
      if (this_diff < 0)
        this_diff = -1.0 * this_diff;
      if (this_diff > THRESHOLD) {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over threshold " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

int main(int argc, char* argv[]) {
  if (argc == 2) {
    block_size = atoi(argv[1]);
  } else {
    block_size = 64;
    cout << "Using default block size = 64\n";
  }

  matrix_A = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
  matrix_B = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
  sequential_C = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
  sequential_opt_C = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
  parallel_C = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
  parallel_opt_C = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];

  for (int i = 0; i < MAT_SIZE; i++) {
    for (int j = 0; j < MAT_SIZE; j++) {
      matrix_A[(i * MAT_SIZE) + j] = 1;
      matrix_B[(i * MAT_SIZE) + j] = 1;
      sequential_C[(i * MAT_SIZE) + j] = 0;
      sequential_opt_C[(i * MAT_SIZE) + j] = 0;
      parallel_C[(i * MAT_SIZE) + j] = 0;
      parallel_opt_C[(i * MAT_SIZE) + j] = 0;
    }
  }
  pthread_t thread_arr[NUM_THREADS];
  pthread_t thread_arr2[NUM_THREADS];

  double clkbegin, clkend;

  clkbegin = rtclock();
  sequential_matmul();
  clkend = rtclock();
  cout << "Time for Sequential version: " << (clkend - clkbegin) << "seconds.\n";

  clkbegin = rtclock();

  for (int i = 0; i < NUM_THREADS; i++) {
    int *arg = (int*) malloc(sizeof(int));
    *arg = i;
    pthread_create(&thread_arr[i], NULL, parallel_work, arg);
  }

  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(thread_arr[i], NULL);
  }
  
  clkend = rtclock();
  cout << "Time for parallel version: " << (clkend - clkbegin) << "seconds.\n";


  clkbegin = rtclock();
  sequential_matmul_opt();
  clkend = rtclock();
  cout << "Time for Sequential Optimized version: " << (clkend - clkbegin) << "seconds.\n";

  clkbegin = rtclock();
  for (int i = 0; i < NUM_THREADS; i++) {
    int *arg = (int*) malloc(sizeof(int));
    *arg = i;
    pthread_create(&thread_arr2[i], NULL, parallel_matmul_opt, arg);
  }

  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(thread_arr2[i], NULL);
  }
  clkend = rtclock();
  cout << "Time for Parallel Optimized version: " << (clkend - clkbegin) << "seconds.\n";

  check_result(sequential_C, parallel_C);
  check_result(sequential_C, sequential_opt_C);
  check_result(sequential_C, parallel_opt_C);
  return 0;
}
