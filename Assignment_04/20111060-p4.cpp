// Compile: g++ -std=c++11 20111060-p4.cpp -o 20111060-p4 -ltbb

#include <cassert>
#include <chrono>
#include <iostream>
#include <tbb/tbb.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

#define N (1 << 26)

uint32_t serial_find_max(const uint32_t* a) {
  uint32_t value_of_max = 0;
  uint32_t index_of_max = -1;
  for (uint32_t i = 0; i < N; i++) {
    uint32_t value = a[i];
    if (value > value_of_max) {
      value_of_max = value;
      index_of_max = i;
    }
  }
  return index_of_max;
}

class MaxFinder{
 
 public:
  const uint32_t* array_copy;
  uint32_t part_max_val;
  uint32_t part_max_ind;

  MaxFinder(const uint32_t* a) : array_copy(a), part_max_val(0), part_max_ind(-1) {}
  
  MaxFinder(MaxFinder& mf, tbb::split) : array_copy(mf.array_copy), part_max_val(0), part_max_ind(-1) {}


  void operator()(const tbb::blocked_range<size_t> &iter){
    uint32_t maxv = part_max_val;
    uint32_t maxi = part_max_ind;
   for (uint32_t i = iter.begin(); i!=iter.end(); i++) {
    uint32_t value = array_copy[i];    
    if (value > maxv) {
      maxv = value;
      maxi = i;
    }
    if(value==maxv && i<maxi) {
      maxi = i;
    }
   }
   part_max_val = maxv;
   part_max_ind = maxi;
  }

  void join(const MaxFinder& max_ob){
   if(max_ob.part_max_val>part_max_val){
    part_max_val = max_ob.part_max_val;
    part_max_ind = max_ob.part_max_ind;
   }
   else if(max_ob.part_max_val==part_max_val && max_ob.part_max_ind<part_max_ind){
    part_max_ind = max_ob.part_max_ind;
   }
  }

};

uint32_t tbb_find_max(const uint32_t* a) {
  // TODO: Implement a parallel max function with Intel TBB
  MaxFinder maxob(a);
  parallel_reduce(tbb::blocked_range<size_t>(0,N),maxob);
  return maxob.part_max_ind;
}

int main() {
  uint32_t* a = new uint32_t[N];
  for (uint32_t i = 0; i < N; i++) {
    a[i] = i;
  }

  HRTimer start = HR::now();
  uint64_t s_max_idx = serial_find_max(a);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Sequential max index: " << s_max_idx << " in " << duration << " us" << endl;

  start = HR::now();
  uint64_t tbb_max_idx = tbb_find_max(a);
  end = HR::now();
  assert(s_max_idx == tbb_max_idx);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel (TBB) max index in " << duration << " us" << endl;

  return EXIT_SUCCESS;
}
