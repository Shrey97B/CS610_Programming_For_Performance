// g++ -std=c++11 -fopenmp 20111060-p3.cpp -o 20111060-p3 -ltbb

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <tbb/tbb.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

const int NUM_INTERVALS = std::numeric_limits<int>::max();

double serial_pi() {
  double dx = 1.0 / NUM_INTERVALS;
  double sum = 0.0;
  for (int i = 0; i < NUM_INTERVALS; ++i) {
    double x = (i + 0.5) * dx;
    double h = std::sqrt(1 - x * x);
    sum += h * dx;
  }
  double pi = 4 * sum;
  return pi;
}

double omp_pi() {
  // TODO: Implement OpenMP version with minimal false sharing

  double dx_omp = 1.0 / NUM_INTERVALS;
  double sum_omp = 0.0;
  #pragma omp parallel for firstprivate(dx_omp) reduction(+:sum_omp)
  for(int i=0;i<NUM_INTERVALS;i++){
    double x_omp = (((double)i) + 0.5)*dx_omp;
    sum_omp += std::sqrt(1 - x_omp*x_omp);
  }

  double pi_omp = ((double)4)*sum_omp*dx_omp;
  return pi_omp;

}

class PiCalculator{
 
 public:
  double part_sum;
  PiCalculator(PiCalculator& pic, tbb::split) : part_sum(0.0) {}
  PiCalculator() : part_sum(0.0) {}

  void operator()(const tbb::blocked_range<size_t> &iter){
    double dx_tbb = 1.0 / NUM_INTERVALS;
    double sum = part_sum;
    for(size_t i =iter.begin();i!=iter.end();i++){
     double x_tbb = (((double)i) + 0.5)*dx_tbb;
     sum += ((double)4)*(std::sqrt(1 - x_tbb*x_tbb) * dx_tbb);
    }
    part_sum = sum; 
  }

  void join(const PiCalculator& pi_ob){
   part_sum+=pi_ob.part_sum;
  }

};

double tbb_pi() {
  // TODO: Implement TBB version with parallel algorithms
  PiCalculator picalc;
  parallel_reduce(tbb::blocked_range<size_t>(0,NUM_INTERVALS),picalc);
  return picalc.part_sum;
}

int main() {
  cout << std::fixed << std::setprecision(5);

  HRTimer start = HR::now();
  double ser_pi = serial_pi();
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial pi: " << ser_pi << " Serial time: " << duration << " microseconds" << endl;

  start = HR::now();
  double o_pi = omp_pi();
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel pi (OMP): " << o_pi << " Parallel time: " << duration << " microseconds"
       << endl;

  start = HR::now();
  double t_pi = tbb_pi();
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel pi (TBB): " << t_pi << " Parallel time: " << duration << " microseconds"
       << endl;

  return EXIT_SUCCESS;
}

// Local Variables:
// compile-command: "g++ -std=c++11 -fopenmp 20111060-p3.cpp -o 20111060-p3 -ltbb; ./20111060-p3"
// End:
