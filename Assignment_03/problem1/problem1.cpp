// Compile: g++ -mavx2 -O2 -o problem1 problem1.cpp
// Execute: ./problem1

//For Intel ICC Compilation: icc -mavx2 -O2 -o problem1 problem1.cpp

#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <immintrin.h>
#include<stdint.h>

using namespace std;

const int N = 1 << 13;
const int Niter = 10;
const double THRESHOLD = 0.00001;
const int unroll_sizei = 8;
const int unroll_sizej = 4;
const int unroll_size = 4;
const int block_size = 256;

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

void reference(double** A, double* x, double* y_ref, double* z_ref) {
  int i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      y_ref[j] = y_ref[j] + A[i][j] * x[i];
      z_ref[j] = z_ref[j] + A[j][i] * x[i];
    }
  }
}

void check_result(double* w_ref, double* w_opt) {
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

// TODO: INITIALLY IDENTICAL TO REFERENCE; MAKE YOUR CHANGES TO OPTIMIZE THIS CODE
// You can create multiple versions of the optimized() function to test your changes
void optimized(double** A, double* x, double* y_opt, double* z_opt) {
  int i, j, it, jt;

  for(it=0;it<N;it+=block_size){
    for(jt=0;jt<N;jt+=block_size){
	  for (i = it; i < (it+block_size); i+=unroll_size) {
	    for (j = jt; j < (jt+block_size); j+=unroll_size) {
	      y_opt[j] = y_opt[j] + A[i][j] * x[i] + A[i+1][j] * x[i+1] + A[i+2][j] * x[i+2] + A[i+3][j] * x[i+3];
	      y_opt[j+1] = y_opt[j+1] + A[i][j+1] * x[i] + A[i+1][j+1] * x[i+1] + A[i+2][j+1] * x[i+2] + A[i+3][j+1] * x[i+3];
	      y_opt[j+2] = y_opt[j+2] + A[i][j+2] * x[i] + A[i+1][j+2] * x[i+1] + A[i+2][j+2] * x[i+2] + A[i+3][j+2] * x[i+3];
	      y_opt[j+3] = y_opt[j+3] + A[i][j+3] * x[i] + A[i+1][j+3] * x[i+1] + A[i+2][j+3] * x[i+2] + A[i+3][j+3] * x[i+3];
	    }
	  }
    }
  }

  for(jt=0;jt<N;jt+=block_size){
    for(it=0;it<N;it+=block_size){
	  for (j = jt; j < (jt+block_size); j+=unroll_size) {
	    for (i = it; i < (it+block_size); i+=unroll_size) {
	      z_opt[j] = z_opt[j] + A[j][i] * x[i] + A[j][i+1] * x[i+1] + A[j][i+2] * x[i+2] + A[j][i+3] * x[i+3];
	      z_opt[j+1] = z_opt[j+1] + A[j+1][i] * x[i] + A[j+1][i+1] * x[i+1] + A[j+1][i+2] * x[i+2] + A[j+1][i+3] * x[i+3];
	      z_opt[j+2] = z_opt[j+2] + A[j+2][i] * x[i] + A[j+2][i+1] * x[i+1] + A[j+2][i+2] * x[i+2] + A[j+2][i+3] * x[i+3];
	      z_opt[j+3] = z_opt[j+3] + A[j+3][i] * x[i] + A[j+3][i+1] * x[i+1] + A[j+3][i+2] * x[i+2] + A[j+3][i+3] * x[i+3];
	    }
	  }
    }
  }
}

void old_optimized(double** A, double* x, double* y_opt, double* z_opt) {
  int i, j, it, jt;

  for(it=0;it<N;it+=block_size){
    for(jt=0;jt<N;jt+=block_size){
	  for (i = it; i < (it+block_size); i+=unroll_sizei) {
	    for (j = jt; j < (jt+block_size); j+=unroll_sizej) {
	      y_opt[j] = y_opt[j] + A[i][j] * x[i] + A[i+1][j] * x[i+1] + A[i+2][j] * x[i+2] + A[i+3][j] * x[i+3] + A[i+4][j] * x[i+4] + A[i+5][j] * x[i+5] + A[i+6][j] * x[i+6] + A[i+7][j] * x[i+7];
	      y_opt[j+1] = y_opt[j+1] + A[i][j+1] * x[i] + A[i+1][j+1] * x[i+1] + A[i+2][j+1] * x[i+2] + A[i+3][j+1] * x[i+3] + A[i+4][j+1] * x[i+4] + A[i+5][j+1] * x[i+5] + A[i+6][j+1] * x[i+6] + A[i+7][j+1] * x[i+7];
	      y_opt[j+2] = y_opt[j+2] + A[i][j+2] * x[i] + A[i+1][j+2] * x[i+1] + A[i+2][j+2] * x[i+2] + A[i+3][j+2] * x[i+3] + A[i+4][j+2] * x[i+4] + A[i+5][j+2] * x[i+5] + A[i+6][j+2] * x[i+6] + A[i+7][j+2] * x[i+7];
	      y_opt[j+3] = y_opt[j+3] + A[i][j+3] * x[i] + A[i+1][j+3] * x[i+1] + A[i+2][j+3] * x[i+2] + A[i+3][j+3] * x[i+3] + A[i+4][j+3] * x[i+4] + A[i+5][j+3] * x[i+5] + A[i+6][j+3] * x[i+6] + A[i+7][j+3] * x[i+7];
	    }
	  }
    }
  }

  for(jt=0;jt<N;jt+=block_size){
    for(it=0;it<N;it+=block_size){
	  for (j = jt; j < (jt+block_size); j+=unroll_sizej) {
	    for (i = it; i < (it+block_size); i+=unroll_sizei) {
	      z_opt[j] = z_opt[j] + A[j][i] * x[i] + A[j][i+1] * x[i+1] + A[j][i+2] * x[i+2] + A[j][i+3] * x[i+3] + A[j][i+4] * x[i+4] + A[j][i+5] * x[i+5] + A[j][i+6] * x[i+6] + A[j][i+7] * x[i+7];
	      z_opt[j+1] = z_opt[j+1] + A[j+1][i] * x[i] + A[j+1][i+1] * x[i+1] + A[j+1][i+2] * x[i+2] + A[j+1][i+3] * x[i+3] + A[j+1][i+4] * x[i+4] + A[j+1][i+5] * x[i+5] + A[j+1][i+6] * x[i+6] + A[j+1][i+7] * x[i+7];
	      z_opt[j+2] = z_opt[j+2] + A[j+2][i] * x[i] + A[j+2][i+1] * x[i+1] + A[j+2][i+2] * x[i+2] + A[j+2][i+3] * x[i+3] + A[j+2][i+4] * x[i+4] + A[j+2][i+5] * x[i+5] + A[j+2][i+6] * x[i+6] + A[j+2][i+7] * x[i+7];
	      z_opt[j+3] = z_opt[j+3] + A[j+3][i] * x[i] + A[j+3][i+1] * x[i+1] + A[j+3][i+2] * x[i+2] + A[j+3][i+3] * x[i+3] + A[j+3][i+4] * x[i+4] + A[j+3][i+5] * x[i+5] + A[j+3][i+6] * x[i+6] + A[j+3][i+7] * x[i+7];
	    }
	  }
    }
  }
}

// You can create multiple versions of the optimized() function to test your changes
void optimized_intrinsic(double** A, double* x, double* y_opt, double* z_opt) {
  int i, j, it, jt;
  
  __m256d rZ, rY, rX1, rX2, rX3, rX4, rX5, rX6, rX7, rX8, rA1, rA2, rA3, rA4, rA5, rA6, rA7, rA8;
  for(it=0;it<N;it+=block_size){
    for(jt=0;jt<N;jt+=block_size){
	  for (i = it; i < (it+block_size); i+=unroll_sizei) {
      rX1 = _mm256_set_pd(x[i],x[i],x[i],x[i]);
      rX2 = _mm256_set_pd(x[i+1],x[i+1],x[i+1],x[i+1]);
      rX3 = _mm256_set_pd(x[i+2],x[i+2],x[i+2],x[i+2]);
      rX4 = _mm256_set_pd(x[i+3],x[i+3],x[i+3],x[i+3]);
      rX5 = _mm256_set_pd(x[i+4],x[i+4],x[i+4],x[i+4]);
      rX6 = _mm256_set_pd(x[i+5],x[i+5],x[i+5],x[i+5]);
      rX7 = _mm256_set_pd(x[i+6],x[i+6],x[i+6],x[i+6]);
      rX8 = _mm256_set_pd(x[i+7],x[i+7],x[i+7],x[i+7]);

	    for (j = jt; j < (jt+block_size); j+=unroll_sizej) {
        rY = _mm256_loadu_pd(&y_opt[j]);
        rA1 = _mm256_loadu_pd(&A[i][j]);
        rA2 = _mm256_loadu_pd(&A[i+1][j]);
        rA3 = _mm256_loadu_pd(&A[i+2][j]);
        rA4 = _mm256_loadu_pd(&A[i+3][j]);
        rA5 = _mm256_loadu_pd(&A[i+4][j]);
        rA6 = _mm256_loadu_pd(&A[i+5][j]);
        rA7 = _mm256_loadu_pd(&A[i+6][j]);
        rA8 = _mm256_loadu_pd(&A[i+7][j]);

        rA1 = _mm256_mul_pd(rA1,rX1);
        rA2 = _mm256_mul_pd(rA2,rX2);
        rA3 = _mm256_mul_pd(rA3,rX3);
        rA4 = _mm256_mul_pd(rA4,rX4);
        rA5 = _mm256_mul_pd(rA5,rX5);
        rA6 = _mm256_mul_pd(rA6,rX6);
        rA7 = _mm256_mul_pd(rA7,rX7);
        rA8 = _mm256_mul_pd(rA8,rX8);

        rA1 = _mm256_add_pd(rA1,rA2);
        rA3 = _mm256_add_pd(rA3,rA4);
        rA5 = _mm256_add_pd(rA5,rA6);
        rA7 = _mm256_add_pd(rA7,rA8);

        rA1 = _mm256_add_pd(rA1,rA3);
        rA5 = _mm256_add_pd(rA5,rA7);

        rA1 = _mm256_add_pd(rA1,rA5);

        rY = _mm256_add_pd(rY,rA1);
        _mm256_storeu_pd(&y_opt[j],rY);

	    }
	  }
    }
  }

  for(jt=0;jt<N;jt+=block_size){
    for(it=0;it<N;it+=block_size){
	  for (j = jt; j < (jt+block_size); j+=unroll_sizej) {
      rZ = _mm256_loadu_pd(&z_opt[j]);
	    for (i = it; i < (it+block_size); i+=unroll_sizei) {

        rX1 = _mm256_loadu_pd(&x[i]);
        rX2 = _mm256_loadu_pd(&x[i+4]);
        rA1 = _mm256_loadu_pd(&A[j][i]);
        rA2 = _mm256_loadu_pd(&A[j][i+4]);
        rA3 = _mm256_loadu_pd(&A[j+1][i]);
        rA4 = _mm256_loadu_pd(&A[j+1][i+4]);
        rA5 = _mm256_loadu_pd(&A[j+2][i]);
        rA6 = _mm256_loadu_pd(&A[j+2][i+4]);
        rA7 = _mm256_loadu_pd(&A[j+3][i]);
        rA8 = _mm256_loadu_pd(&A[j+3][i+4]);

        rA1 = _mm256_mul_pd(rA1,rX1);  
        rA2 = _mm256_mul_pd(rA2,rX2);
        rA3 = _mm256_mul_pd(rA3,rX1);
        rA4 = _mm256_mul_pd(rA4,rX2);
        rA5 = _mm256_mul_pd(rA5,rX1);
        rA6 = _mm256_mul_pd(rA6,rX2);
        rA7 = _mm256_mul_pd(rA7,rX1);
        rA8 = _mm256_mul_pd(rA8,rX2);

        rA1 = _mm256_hadd_pd(rA1,rA2);  //(temp[j][i:i+1],temp[j][i+4:i+5],temp[j][i+2:i+3],temp[j][i+6:i+7])
        rA5 = _mm256_hadd_pd(rA5,rA6);  //(temp[j+2][i:i+1],temp[j+2][i+4:i+5],temp[j+2][i+2:i+3],temp[j+2][i+6:i+7])
        rA1 = _mm256_hadd_pd(rA1,rA5);  //(temp[j][i:i+1] + temp[j][i+4:i+5],temp[j+2][i:i+1] + temp[j+2][i+4:i+5],temp[j][i+2:i+3]+temp[j][i+6:i+7],temp[j+2][i+2:i+3]+temp[j+2][i+6:i+7])
        rA1 = _mm256_permute4x64_pd(rA1,0xD8); //(temp[j][i:i+1] + temp[j][i+4:i+5],temp[j][i+2:i+3]+temp[j][i+6:i+7],temp[j+2][i:i+1] + temp[j+2][i+4:i+5],temp[j+2][i+2:i+3]+temp[j+2][i+6:i+7])

        rA3 = _mm256_hadd_pd(rA3,rA4);  //(temp[j+1][i:i+1],temp[j+1][i+4:i+5],temp[j+1][i+2:i+3],temp[j+1][i+6:i+7])
        rA7 = _mm256_hadd_pd(rA5,rA6);  //(temp[j+4][i:i+1],temp[j+4][i+4:i+5],temp[j+4][i+2:i+3],temp[j+4][i+6:i+7])
        rA3 = _mm256_hadd_pd(rA3,rA7);  //(temp[j+1][i:i+1] + temp[j+1][i+4:i+5],temp[j+4][i:i+1] + temp[j+4][i+4:i+5],temp[j+1][i+2:i+3]+temp[j+1][i+6:i+7],temp[j+4][i+2:i+3]+temp[j+4][i+6:i+7])
        rA3 = _mm256_permute4x64_pd(rA3,0xD8);  //(temp[j+1][i:i+1] + temp[j+1][i+4:i+5],temp[j+1][i+2:i+3]+temp[j+1][i+6:i+7],temp[j+4][i:i+1] + temp[j+4][i+4:i+5],temp[j+4][i+2:i+3]+temp[j+4][i+6:i+7])

        rA1 = _mm256_hadd_pd(rA1,rA3);
        rZ = _mm256_add_pd(rZ,rA1);
      }

      _mm256_storeu_pd(&z_opt[j],rZ);

	  }
    }
  }
  

}

// You can create multiple versions of the optimized() function to test your changes
/* Better version implemented above
void optimized_intrinsic2(double** A, double* x, double* y_opt, double* z_opt) {
  int i, j, it, jt;
  
  __m256d rZ, rY, rX1, rX2, rX3, rX4, rX5, rX6, rX7, rX8, rA1, rA2, rA3, rA4, rA5, rA6, rA7, rA8;
  for(it=0;it<N;it+=block_size){
    for(jt=0;jt<N;jt+=block_size){
	  for (i = it; i < (it+block_size); i+=unroll_size) {
      rX1 = _mm256_set_pd(x[i],x[i],x[i],x[i]);
      rX2 = _mm256_set_pd(x[i+1],x[i+1],x[i+1],x[i+1]);
      rX3 = _mm256_set_pd(x[i+2],x[i+2],x[i+2],x[i+2]);
      rX4 = _mm256_set_pd(x[i+3],x[i+3],x[i+3],x[i+3]);


	    for (j = jt; j < (jt+block_size); j+=unroll_size) {
        rY = _mm256_loadu_pd(&y_opt[j]);
        rA1 = _mm256_loadu_pd(&A[i][j]);
        rA2 = _mm256_loadu_pd(&A[i+1][j]);
        rA3 = _mm256_loadu_pd(&A[i+2][j]);
        rA4 = _mm256_loadu_pd(&A[i+3][j]);

        rA1 = _mm256_mul_pd(rA1,rX1);
        rA2 = _mm256_mul_pd(rA2,rX2);
        rA3 = _mm256_mul_pd(rA3,rX3);
        rA4 = _mm256_mul_pd(rA4,rX4);

        rA1 = _mm256_add_pd(rA1,rA2);
        rA3 = _mm256_add_pd(rA3,rA4);
        rA1 = _mm256_add_pd(rA1,rA3);

        rA1 = _mm256_add_pd(rA1,rA5);

        rY = _mm256_add_pd(rY,rA1);
        _mm256_storeu_pd(&y_opt[j],rY);

	    }
	  }
    }
  }

  for(jt=0;jt<N;jt+=block_size){
    for(it=0;it<N;it+=block_size){
	  for (j = jt; j < (jt+block_size); j+=unroll_size) {
      rZ = _mm256_loadu_pd(&z_opt[j]);
	    for (i = it; i < (it+block_size); i+=unroll_size) {

        rX1 = _mm256_loadu_pd(&x[i]);
        rA1 = _mm256_loadu_pd(&A[j][i]);
        rA3 = _mm256_loadu_pd(&A[j+1][i]);
        rA5 = _mm256_loadu_pd(&A[j+2][i]);
        rA7 = _mm256_loadu_pd(&A[j+3][i]);

        rA1 = _mm256_mul_pd(rA1,rX1);  
        rA3 = _mm256_mul_pd(rA3,rX1);
        rA5 = _mm256_mul_pd(rA5,rX1);
        rA7 = _mm256_mul_pd(rA7,rX1);


        rA1 = _mm256_hadd_pd(rA1,rA5);  //(temp[j][i:i+1] + temp[j][i+4:i+5],temp[j+2][i:i+1] + temp[j+2][i+4:i+5],temp[j][i+2:i+3]+temp[j][i+6:i+7],temp[j+2][i+2:i+3]+temp[j+2][i+6:i+7])
        rA1 = _mm256_permute4x64_pd(rA1,0xD8); //(temp[j][i:i+1] + temp[j][i+4:i+5],temp[j][i+2:i+3]+temp[j][i+6:i+7],temp[j+2][i:i+1] + temp[j+2][i+4:i+5],temp[j+2][i+2:i+3]+temp[j+2][i+6:i+7])

        rA3 = _mm256_hadd_pd(rA3,rA7);  //(temp[j+1][i:i+1] + temp[j+1][i+4:i+5],temp[j+4][i:i+1] + temp[j+4][i+4:i+5],temp[j+1][i+2:i+3]+temp[j+1][i+6:i+7],temp[j+4][i+2:i+3]+temp[j+4][i+6:i+7])
        rA3 = _mm256_permute4x64_pd(rA3,0xD8);  //(temp[j+1][i:i+1] + temp[j+1][i+4:i+5],temp[j+1][i+2:i+3]+temp[j+1][i+6:i+7],temp[j+4][i:i+1] + temp[j+4][i+4:i+5],temp[j+4][i+2:i+3]+temp[j+4][i+6:i+7])

        rA1 = _mm256_hadd_pd(rA1,rA3);
        rZ = _mm256_add_pd(rZ,rA1);
      }

      _mm256_storeu_pd(&z_opt[j],rZ);

	  }
    }
  }
  

}
*/

int main() {
  double clkbegin, clkend;
  double t;

  int i, j, it;
  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(5);

  double** A;
  A = new double*[N];
  for (int i = 0; i < N; i++) {
    A[i] = new double[N];
  }

  double *x, *y_ref, *z_ref, *y_opt, *z_opt;
  x = new double[N];
  y_ref = new double[N];
  z_ref = new double[N];
  y_opt = new double[N];
  z_opt = new double[N];

  for (i = 0; i < N; i++) {
    x[i] = i;
    y_ref[i] = 1.0;
    y_opt[i] = 1.0;
    z_ref[i] = 2.0;
    z_opt[i] = 2.0;
    for (j = 0; j < N; j++) {
      A[i][j] = (i + 2.0 * j) / (2.0 * N);
    }
  }

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    reference(A, x, y_ref, z_ref);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Reference Version: Matrix Size = " << N << ", " << 4.0 * 1e-9 * N * N * Niter / t
       << " GFLOPS; Time = " << t / Niter << " sec\n";

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    old_optimized(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Old Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);

  // Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // Another optimized version possibly

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    optimized(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);

  // Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  clkbegin = rtclock();
 for (it = 0; it < Niter; it++) {
    optimized_intrinsic(A, x, y_opt, z_opt);
 }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Intrinsic Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);

/*
  // Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  //optimized_intrinsic works better than the below version
  clkbegin = rtclock();
 for (it = 0; it < Niter; it++) {
    optimized_intrinsic2(A, x, y_opt, z_opt);
 }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Intrinsic Version 2: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);
*/


  return EXIT_SUCCESS;
}
