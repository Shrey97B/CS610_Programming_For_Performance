// Compile: g++ -std=c++11 -fopenmp 20111060-p1.cpp -o 20111060-p1 -ltbb

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include "tbb/tbb.h"

#define N 50

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

long fibt2[N+1];
omp_lock_t locks[N+1];

// Serial Fibonacci

long ser_fib(int n) {
    
  if (n == 0 || n == 1) {
    return (n);
  }
  return ser_fib(n - 1) + ser_fib(n - 2);
}

long execTaskFib1(int num){
    
  if(num<2){
    return num;
  }
    
  if(num<=N/2){
      return execTaskFib1(num-1) + execTaskFib1(num-2);
  }

  long num1,num2;
  

   #pragma omp task shared(num1) firstprivate(num)
   {
   num1 = execTaskFib1(num-1);
   }

   #pragma omp task shared(num2) firstprivate(num)
   {
    num2 = execTaskFib1(num-2);
   }
  
  #pragma omp taskwait

  return num1 + num2;
}

long omp_fib_v1(int n) {
  // TODO: Implement OpenMP version with explicit tasks

  long ans;
  #pragma omp parallel shared(ans)
  {
    #pragma omp single
    {
      ans = execTaskFib1(n);
    }
  }

  return ans;

}

long execTaskFib2(int num){
    
  if(num<2){
    return num;
  }

  bool flag = false;
  omp_set_lock(&locks[num]);
      if(fibt2[num]!=-1){
          flag = true;
      }
  omp_unset_lock(&locks[num]);
    
  if(flag){
      return fibt2[num];
  }
  
/*
  if(num<=25){
      return execTaskFib2(num-1) + execTaskFib2(num-2);
  }
*/

  long num1,num2;
  

   #pragma omp task shared(num1) firstprivate(num)
   {
   num1 = execTaskFib2(num-1);
   }

   #pragma omp task shared(num2) firstprivate(num)
   {
    num2 = execTaskFib2(num-2);
   }
  
  #pragma omp taskwait
    
    omp_set_lock(&locks[num]);
        fibt2[num] = num1 + num2;
    omp_unset_lock(&locks[num]);

  return num1 + num2;
}


long omp_fib_v2(int n) {
  // TODO: Implement an optimized OpenMP version with any valid optimization

  for(int i=0;i<=N;i++){
      fibt2[i] = -1;
      omp_init_lock(&locks[i]);
  }
    
  long ans;
  #pragma omp parallel shared(ans)
  {
    #pragma omp single
    {
      ans = execTaskFib2(n);
    }
  }

  for(int i=0;i<=N;i++){
      omp_destroy_lock(&locks[i]);
  }

  return ans;

}

class BlockFibT: public tbb::task{

public:    
    const long num;
    long* const fib;
    
    BlockFibT(long n,long* ans):num(n),fib(ans){
    }

    long serialFib(long n){
	if(n<2){
	  return n;
	}
	return serialFib(n-1) + serialFib(n-2);
    }
    
    tbb::task* execute(){
        if(num<2){
            *fib = num;
            return NULL;
        }

	if(num<=N/2){
	    *fib = serialFib(num);
	    return NULL;
	}
        
        long num1,num2;
        BlockFibT& t1 = *new(tbb::task::allocate_child()) BlockFibT(num-1,&num1);
        BlockFibT& t2 = *new(tbb::task::allocate_child()) BlockFibT(num-2,&num2);
        set_ref_count(3);
        spawn(t1);
        spawn_and_wait_for_all(t2);
        *fib = num1 + num2;
        
        return NULL;
        
    }
    
};


long pbfib(int n){
    if(n<2){
        return n;
    }
    
    if(n<=N/2){
        return pbfib(n-1) + pbfib(n-2);
    }
    
    long n1,n2;
    tbb::task_group fibtg;
    fibtg.run([&]{n1 = pbfib(n-1);});
    fibtg.run([&]{n2 = pbfib(n-2);});
    fibtg.wait();
    
    return n1+n2;
    
}

long tbb_fib_blocking(int n) {
  // TODO: Implement Intel TBB version with blocking style
  long ans = 0;
  BlockFibT& maint = *new(tbb::task::allocate_root()) BlockFibT(n,&ans);
  tbb::task::spawn_root_and_wait(maint);
  return ans;

/*
  tbb::task_group fibtg;
  fibtg.run([&]{ans = pbfib(n);});
  fibtg.wait();
    
  return ans;
*/
  
}


class FibCoTask: public tbb::task{

public:    
    long* const fib;
    long num1,num2;
    FibCoTask(long* ans):fib(ans){
    }

    tbb::task* execute(){
	*fib = num1 + num2;
	return NULL;
    }

};

class CPassFibT: public tbb::task{

public:    
    const long num;
    long* const fib;
    
    CPassFibT(long n,long* ans):num(n),fib(ans){
    }
    
    long serialFib(long n){
	if(n<2){
	  return n;
	}
	return serialFib(n-1) + serialFib(n-2);
    }

    tbb::task* execute(){
        if(num<2){
            *fib = num;
            return NULL;
        }

	if(num<=N/2){
	    *fib = serialFib(num);
	    return NULL;
	}
        
	
	FibCoTask& ct = *new(allocate_continuation()) FibCoTask(fib);
        CPassFibT& t1 = *new(ct.allocate_child()) CPassFibT(num-1,&ct.num1);
        CPassFibT& t2 = *new(ct.allocate_child()) CPassFibT(num-2,&ct.num2);
        ct.set_ref_count(2);
        spawn(t1);
        spawn(t2);
        
        return NULL;
        
    }
    
};

long tbb_fib_cps(int n) {
  // TODO: Implement Intel TBB version with continuation passing style
  long ans = 0;
  CPassFibT& maint = *new(tbb::task::allocate_root()) CPassFibT(n,&ans);
  tbb::task::spawn_root_and_wait(maint);
  return ans;
}

int main(int argc, char** argv) {
  cout << std::fixed << std::setprecision(5);

  HRTimer start = HR::now();
  long s_fib = ser_fib(N);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial time: " << duration << " microseconds" << endl;

  start = HR::now();
  long omp_v1 = omp_fib_v1(N);
  end = HR::now();
  assert(s_fib == omp_v1);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "OMP v1 time: " << duration << " microseconds" << endl;

  start = HR::now();
  long omp_v2 = omp_fib_v2(N);
  end = HR::now();
  assert(s_fib == omp_v2);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "OMP v2 time: " << duration << " microseconds" << endl;
    
  start = HR::now();
  long blocking_fib = tbb_fib_blocking(N);
  end = HR::now();
  assert(s_fib == blocking_fib);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "TBB (blocking) time: " << duration << " microseconds" << endl;

  start = HR::now();
  long cps_fib = tbb_fib_cps(N);
  end = HR::now();
  assert(s_fib == cps_fib);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "TBB (cps) time: " << duration << " microseconds" << endl;

  return EXIT_SUCCESS;
}
