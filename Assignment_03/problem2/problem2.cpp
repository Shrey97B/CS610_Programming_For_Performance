// Compile: g++ -mavx2 -O2 -o problem2 problem2.cpp
// Execute: ./problem2

//For Intel ICC Compilation: icc -mavx2 -O2 -o problem1 problem1.cpp

#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include<immintrin.h>
#include<string.h>
#include<stdint.h>

using namespace std;

const int N = 1024;
const int Niter = 10;
const double THRESHOLD = 0.0000001;
const int unroll_size=4;
const int block_size = 32;

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << endl;
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void reference(double** A, double** B, double** C) {
  int i, j, k;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      for (k = 0; k < i + 1; k++) {
        C[i][j] += A[k][i] * B[j][k];
      }
    }
  }
}

void check_result(double** w_ref, double** w_opt) {
  double maxdiff, this_diff;
  int numdiffs;
  int i, j;
  numdiffs = 0;
  maxdiff = 0;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      this_diff = w_ref[i][j] - w_opt[i][j];
      if (fabs(this_diff) > THRESHOLD) {
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

// TODO: THIS IS INITIALLY IDENTICAL TO REFERENCE. MAKE YOUR CHANGES TO OPTIMIZE THIS FUNCTION
// You can create multiple versions of the optimized() function to test your changes
void old_optimized(double** A, double** B, double** C) {
  int i, j, k, it, jt, kt;
  double temp[unroll_size] ={0};

for(it=0;it<N;it+=block_size){
 for(jt=0;jt<N;jt+=block_size){
  for (i = it; i < (it+block_size); i++) {
    for (j = jt; j < (jt+block_size); j+=unroll_size) {
      temp[0] = C[i][j];
      temp[1] = C[i][j+1];
      temp[2] = C[i][j+2];
      temp[3] = C[i][j+3];
      for (k = 0; k+unroll_size-1 < i+1 ; k+=unroll_size) {
        temp[0] += A[k][i] * B[j][k] + A[k+1][i] * B[j][k+1] + A[k+2][i] * B[j][k+2] + A[k+3][i] * B[j][k+3];
        temp[1] += A[k][i] * B[j+1][k] + A[k+1][i] * B[j+1][k+1] + A[k+2][i] * B[j+1][k+2] + A[k+3][i] * B[j+1][k+3];
        temp[2] += A[k][i] * B[j+2][k] + A[k+1][i] * B[j+2][k+1] + A[k+2][i] * B[j+2][k+2] + A[k+3][i] * B[j+2][k+3];
        temp[3] += A[k][i] * B[j+3][k] + A[k+1][i] * B[j+3][k+1] + A[k+2][i] * B[j+3][k+2] + A[k+3][i] * B[j+3][k+3];
      }
      for(;k<i+1;k++){
	temp[0] += A[k][i] * B[j][k];
	temp[1] += A[k][i] * B[j+1][k];
	temp[2] += A[k][i] * B[j+2][k];
	temp[3] += A[k][i] * B[j+3][k];
      }
     C[i][j] = temp[0];
     C[i][j+1] = temp[1];
     C[i][j+2] = temp[2];
     C[i][j+3] = temp[3];
    }
   }
  }
 }
}

void optimized(double** A, double** B, double** C) {
  int i, j, k, it, jt, kt;
  double temp[unroll_size] ={0};

for(it=0;it<N;it+=block_size){
 for(jt=0;jt<N;jt+=block_size){
  for(kt=0;kt<it+1;kt+=block_size){
  for (i = it; i < (it+block_size); i++) {
    for (j = jt; j < (jt+block_size); j+=unroll_size) {
      temp[0] = C[i][j];
      temp[1] = C[i][j+1];
      temp[2] = C[i][j+2];
      temp[3] = C[i][j+3];
      for (k = kt; k+unroll_size-1 < min(kt+block_size,i+1) ; k+=unroll_size) {
        temp[0] += A[k][i] * B[j][k] + A[k+1][i] * B[j][k+1] + A[k+2][i] * B[j][k+2] + A[k+3][i] * B[j][k+3];
        temp[1] += A[k][i] * B[j+1][k] + A[k+1][i] * B[j+1][k+1] + A[k+2][i] * B[j+1][k+2] + A[k+3][i] * B[j+1][k+3];
        temp[2] += A[k][i] * B[j+2][k] + A[k+1][i] * B[j+2][k+1] + A[k+2][i] * B[j+2][k+2] + A[k+3][i] * B[j+2][k+3];
        temp[3] += A[k][i] * B[j+3][k] + A[k+1][i] * B[j+3][k+1] + A[k+2][i] * B[j+3][k+2] + A[k+3][i] * B[j+3][k+3];
	
      }

      for(;k<min(kt+block_size,i+1);k++){
	temp[0] += A[k][i] * B[j][k];
	temp[1] += A[k][i] * B[j+1][k];
	temp[2] += A[k][i] * B[j+2][k];
	temp[3] += A[k][i] * B[j+3][k];
      }
     C[i][j] = temp[0];
     C[i][j+1] = temp[1];
     C[i][j+2] = temp[2];
     C[i][j+3] = temp[3];

     }
    }
   }
  }
 }
}

void optimized_without_temp(double** A, double** B, double** C) {
  int i, j, k, it, jt, kt;

for(it=0;it<N;it+=block_size){
 for(jt=0;jt<N;jt+=block_size){
  for(kt=0;kt<it+1;kt+=block_size){
  for (i = it; i < (it+block_size); i++) {
    for (j = jt; j < (jt+block_size); j+=unroll_size) {
      for (k = kt; k+unroll_size-1 < min(kt+block_size,i+1) ; k+=unroll_size) {
        C[i][j] += A[k][i] * B[j][k] + A[k+1][i] * B[j][k+1] + A[k+2][i] * B[j][k+2] + A[k+3][i] * B[j][k+3];
        C[i][j+1] += A[k][i] * B[j+1][k] + A[k+1][i] * B[j+1][k+1] + A[k+2][i] * B[j+1][k+2] + A[k+3][i] * B[j+1][k+3];
        C[i][j+2] += A[k][i] * B[j+2][k] + A[k+1][i] * B[j+2][k+1] + A[k+2][i] * B[j+2][k+2] + A[k+3][i] * B[j+2][k+3];
        C[i][j+3] += A[k][i] * B[j+3][k] + A[k+1][i] * B[j+3][k+1] + A[k+2][i] * B[j+3][k+2] + A[k+3][i] * B[j+3][k+3];
      }
      for(;k<min(kt+block_size,i+1);k++){
	C[i][j] += A[k][i] * B[j][k];
	C[i][j+1] += A[k][i] * B[j+1][k];
	C[i][j+2] += A[k][i] * B[j+2][k];
	C[i][j+3] += A[k][i] * B[j+3][k];
      }
     }
    }
   }
  }
 }
}

void optimized_intrinsic(double** A, double** B, double** C) {
  int i, j, k, it, jt, kt;

for(it=0;it<N;it+=block_size){
 for(jt=0;jt<N;jt+=block_size){
  for(kt=0;kt<it+1;kt+=block_size){
  for (i = it; i < (it+block_size); i++) {
    for (j = jt; j < (jt+block_size); j+=unroll_size) {
	__m256d rC,rB1,rB2,rB3,rB4,rA;
	rC = _mm256_loadu_pd(&C[i][j]);
      for (k = kt; k+unroll_size-1 < min(kt+block_size,i+1) ; k+=unroll_size) {

	rB1 = _mm256_loadu_pd(&B[j][k]);
	rB2 = _mm256_loadu_pd(&B[j+1][k]);
	rB3 = _mm256_loadu_pd(&B[j+2][k]);
	rB4 = _mm256_loadu_pd(&B[j+3][k]);

	rA = _mm256_set_pd(A[k+3][i],A[k+2][i],A[k+1][i],A[k][i]);

	rB1 = _mm256_mul_pd(rA,rB1);
	rB2 = _mm256_mul_pd(rA,rB2);
	rB3 = _mm256_mul_pd(rA,rB3);
	rB4 = _mm256_mul_pd(rA,rB4);

	rB1 = _mm256_hadd_pd(rB1,rB3);
	rB1 = _mm256_permute4x64_pd(rB1,0xD8);
	rB2 = _mm256_hadd_pd(rB2,rB4);
	rB2 = _mm256_permute4x64_pd(rB2,0xD8);

	rB1 = _mm256_hadd_pd(rB1,rB2);
	rC = _mm256_add_pd(rC,rB1);

      }


      for(;k<min(kt+block_size,i+1);k++){
	rA = _mm256_set_pd(A[k][i],A[k][i],A[k][i],A[k][i]);
	rB1 = _mm256_set_pd(B[j+3][k],B[j+2][k],B[j+1][k],B[j][k]);
	rC = _mm256_add_pd(rC,_mm256_mul_pd(rA,rB1));
      }

      _mm256_storeu_pd(&C[i][j],rC);

     }
    }
   }
  }
 }
}


int main() {
  double clkbegin, clkend;
  double t;

  int i, j, it;
  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(5);

  double **A, **B, **C_ref, **C_opt;
  A = new double*[N];
  B = new double*[N];
  C_ref = new double*[N];
  C_opt = new double*[N];
  for (i = 0; i < N; i++) {
    A[i] = new double[N];
    B[i] = new double[N];
    C_ref[i] = new double[N];
    C_opt[i] = new double[N];
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      A[i][j] = i + j + 1;
      B[i][j] = (i + 1) * (j + 1);
      C_ref[i][j] = 0.0;
      C_opt[i][j] = 0.0;
    }
  }

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++)
    reference(A, B, C_ref);
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Reference Version: Matrix Size = " << N << ", " << 2.0 * 1e-9 * N * N * Niter / t
       << " GFLOPS; Time = " << t / Niter << " sec\n";

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++)
    old_optimized(A, B, C_opt);
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Old Optimized Version with ij level blocking: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(C_ref, C_opt);

  // Reset
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C_opt[i][j] = 0.0;
    }
  }

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++)
    optimized(A, B, C_opt);
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version with ijk level blocking: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(C_ref, C_opt);

  // Reset
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C_opt[i][j] = 0.0;
    }
  }

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++)
    optimized_without_temp(A, B, C_opt);
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version without temp: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(C_ref, C_opt);

  // Reset
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C_opt[i][j] = 0.0;
    }
  }

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++)
    optimized_intrinsic(A, B, C_opt);
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version with intrinsic: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(C_ref, C_opt);

  return EXIT_SUCCESS;
}
