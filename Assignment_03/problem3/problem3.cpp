// Compile: g++ -O2 -fopenmp -o problem3 problem3.cpp
// Execute: ./problem3

//For Intel ICC Compilation: icc -fopenmp -O2 -o problem1 problem1.cpp

#include <cassert>
#include <iostream>
#include <omp.h>
#include <cmath>

#define N (1 << 12)
#define ITER 100

using namespace std;

const int unroll_size = 4;
const int block_size = 256;

void check_result(uint32_t** w_ref, uint32_t** w_opt) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      assert(w_ref[i][j] == w_opt[i][j]);
    }
  }
  cout << "No differences found between base and test versions\n";
}

void reference(uint32_t** A) {
  int i, j, k;
  for (k = 0; k < ITER; k++) {
    for (i = 1; i < N; i++) {
      for (j = 0; j < (N - 1); j++) {
        A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
      }
    }
  }
}

// TODO: MAKE YOUR CHANGES TO OPTIMIZE THIS FUNCTION
void omp_version(uint32_t** A) {
//kij loopwith 4 level unrolling over i and j parallelized over j for loop
	
int i, j, k;
  for (k = 0; k < ITER; k++) {
    for (i = 1; i + unroll_size-1< N; i+=unroll_size) {
	
      #pragma omp parallel for shared(A)
      for (j = 0; j < (N - unroll_size); j+=unroll_size) {
        A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
	A[i][j + 2] = A[i - 1][j + 2] + A[i][j + 2];
	A[i][j + 3] = A[i - 1][j + 3] + A[i][j + 3];
	A[i][j + 4] = A[i - 1][j + 4] + A[i][j + 4];
	A[i+1][j + 1] = A[i][j + 1] + A[i+1][j + 1];
	A[i+1][j + 2] = A[i][j + 2] + A[i+1][j + 2];
	A[i+1][j + 3] = A[i][j + 3] + A[i+1][j + 3];
	A[i+1][j + 4] = A[i][j + 4] + A[i+1][j + 4];
        A[i+2][j + 1] = A[i + 1][j + 1] + A[i+2][j + 1];
	A[i+2][j + 2] = A[i + 1][j + 2] + A[i+2][j + 2];
	A[i+2][j + 3] = A[i + 1][j + 3] + A[i+2][j + 3];
	A[i+2][j + 4] = A[i + 1][j + 4] + A[i+2][j + 4];
        A[i+3][j + 1] = A[i + 2][j + 1] + A[i+3][j + 1];
	A[i+3][j + 2] = A[i + 2][j + 2] + A[i+3][j + 2];
	A[i+3][j + 3] = A[i + 2][j + 3] + A[i+3][j + 3];
	A[i+3][j + 4] = A[i + 2][j + 4] + A[i+3][j + 4];
      }
	int f = (N-1)%unroll_size;
	int done = (N-1)/unroll_size;
        j = done*unroll_size;
	for(int x=0;x<f;x++,j++){
	 A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
	 A[i+1][j + 1] = A[i][j + 1] + A[i+1][j + 1];
	 A[i+2][j + 1] = A[i + 1][j + 1] + A[i+2][j + 1];
	 A[i+3][j + 1] = A[i + 2][j + 1] + A[i+3][j + 1];	
	}
    }
   for(;i<N;i++){
	for(j=0;j<N-1;j++){
	 A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
	}
   }
  }
	
}

void omp_version2(uint32_t** A) {
//kji loop with 4 way unrolling over i and j with defined work distribution on j to enable blocking
  int i, j, k ,jt ,it;
  for (k = 0; k < ITER; k++) {

   #pragma omp parallel shared(A)
   {
   double numthr = omp_get_num_threads();
   double thid = omp_get_thread_num();
   int niter = (int)ceil((double)(N-1)/numthr);
   int jbeg = (thid)*niter;
   int jend = min(N-1,jbeg + niter);
   for(jt=jbeg;jt<jend;jt+=block_size){
   for(it=1;it<N;it+=block_size){
   for (j = jt; j+unroll_size-1< min(jend,jt+block_size); j+=unroll_size) {
	
	int f = min(block_size,N-it)%unroll_size;
    for (i = it; i+unroll_size-1< min(N,it+block_size); i+=unroll_size) {
        A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
	A[i+1][j + 1] = A[i][j + 1] + A[i+1][j + 1];
	A[i+2][j + 1] = A[i + 1][j + 1] + A[i+2][j + 1];
	A[i+3][j + 1] = A[i + 2][j + 1] + A[i+3][j + 1];
	A[i][j + 2] = A[i - 1][j + 2] + A[i][j + 2];
	A[i+1][j + 2] = A[i][j + 2] + A[i+1][j + 2];
	A[i+2][j + 2] = A[i + 1][j + 2] + A[i+2][j + 2];
	A[i+3][j + 2] = A[i + 2][j + 2] + A[i+3][j + 2];
        A[i][j + 3] = A[i - 1][j + 3] + A[i][j + 3];
	A[i+1][j + 3] = A[i][j + 3] + A[i+1][j + 3];
	A[i+2][j + 3] = A[i + 1][j + 3] + A[i+2][j + 3];
	A[i+3][j + 3] = A[i + 2][j + 3] + A[i+3][j + 3];
	A[i][j + 4] = A[i - 1][j + 4] + A[i][j + 4];
	A[i+1][j + 4] = A[i][j + 4] + A[i+1][j + 4];
	A[i+2][j + 4] = A[i + 1][j + 4] + A[i+2][j + 4];
	A[i+3][j + 4] = A[i + 2][j + 4] + A[i+3][j + 4];
      }
	for(int x=0;x<f;x++,i++){
	 A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
	 A[i][j + 2] = A[i - 1][j + 2] + A[i][j + 2];
	 A[i][j + 3] = A[i - 1][j + 3] + A[i][j + 3];
	 A[i][j + 4] = A[i - 1][j + 4] + A[i][j + 4];
	}
     }
     for(;j<min(jend,jt+block_size);j++){
	for(i=it;i<min(N,it+block_size);i++){
		A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
	}
     }
    }
   }
  }
  }
}

void omp_version3(uint32_t** A) {
//kij loop with 4 way unrolling over j with defined work distribution on j but not using for to avoid multiple thread spawns in inner loop
  int i, j, k;
  for (k = 0; k < ITER; k++) {
   #pragma omp parallel shared(A)
    {
   double numthr = omp_get_num_threads();
   double thid = omp_get_thread_num();
   int niter = (int)ceil((double)(N-1)/numthr);
   int jbeg = (thid)*niter;
   int jend = min(N-1,jbeg + niter);
    for (i = 1; i<N; i++) {
      for (j = jbeg; j+3< jend; j+=4) {
        A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
	A[i][j + 2] = A[i - 1][j + 2] + A[i][j + 2];
	A[i][j + 3] = A[i - 1][j + 3] + A[i][j + 3];
	A[i][j + 4] = A[i - 1][j + 4] + A[i][j + 4];
      }
      for(;j<jend;j++){
	A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
      }
    }
   }
  }
}

/*
void omp_version4(uint32_t** A) {

//jki loop with blocking but no unrolling with defined work distribution over j to enable blocking
  int i, j, k ,jt ,it;

   #pragma omp parallel shared(A)
   {
   double numthr = omp_get_num_threads();
   double thid = omp_get_thread_num();
   int niter = (int)ceil((double)(N-1)/numthr);
   int jbeg = (thid)*niter;
   int jend = min(N-1,jbeg + niter);
   for(jt=jbeg;jt<jend;jt+=block_size){
   for(k=0;k<ITER;k++){
   for(it=1;it<N;it+=block_size){
   for (j = jt; j< min(jend,jt+block_size); j++) {
    for (i = it; i< min(N,it+block_size); i++) {
        A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
      }
     }
    }
   }
  }
  }
}
*/

int main() {
  uint32_t** A_ref = new uint32_t*[N];
  for (int i = 0; i < N; i++) {
    A_ref[i] = new uint32_t[N];
  }

  uint32_t** A_omp = new uint32_t*[N];
  for (int i = 0; i < N; i++) {
    A_omp[i] = new uint32_t[N];
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_ref[i][j] = i + j + 1;
      A_omp[i][j] = i + j + 1;
    }
  }

  double start = omp_get_wtime();
  reference(A_ref);
  double end = omp_get_wtime();
  cout << "Time for reference version: " << end - start << " seconds\n";

  start = omp_get_wtime();
  omp_version(A_omp);
  end = omp_get_wtime();
  check_result(A_ref, A_omp);
  cout << "Version1: Time with OpenMP: " << end - start << " seconds\n";

  // Reset
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_omp[i][j] = i + j + 1;
    }
  }

  // Another optimized version possibly
  start = omp_get_wtime();
  omp_version2(A_omp);
  end = omp_get_wtime();
  check_result(A_ref, A_omp);
  cout << "Version2: Time with OpenMP: " << end - start << " seconds\n";

  // Reset
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_omp[i][j] = i + j + 1;
    }
  }

  // Another optimized version possibly
  start = omp_get_wtime();
  omp_version3(A_omp);
  end = omp_get_wtime();
  check_result(A_ref, A_omp);
  cout << "Version3: Time with OpenMP: " << end - start << " seconds\n";

  // The below code implements version 4 but is not fruitful. Hence is is commented
/*
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_omp[i][j] = i + j + 1;
    }
  }

  start = omp_get_wtime();
  omp_version4(A_omp);
  end = omp_get_wtime();
  check_result(A_ref, A_omp);
  cout << "Version4: Time with OpenMP: " << end - start << " seconds. Blocking does not optimize with jki version\n";
*/

  return EXIT_SUCCESS;
}
