#ifndef KERNELPLAN_H
#define KERNELPLAN_H

class KernelPlan
{
public:
  // default constructor
  KernelPlan() : grid(1,1), threads(1,1), shared_size(0), num_blocks(0),
		 num_virt_part(0) {}
  // constructor taking initial values
  KernelPlan(dim3 grid, dim3 threads, size_t shared_size,
	     unsigned int num_blocks, unsigned int num_virt_part
	     ): grid(grid), threads(threads), shared_size(shared_size),
             num_blocks(num_blocks), num_virt_part(num_virt_part) {}
  // assignment operator
  KernelPlan& operator= (const KernelPlan &kp) {
    if(this != &kp) {
      grid = kp.grid;
      threads = kp.threads;
      shared_size = kp.shared_size;
      num_blocks = kp.num_blocks;
      num_virt_part = kp.num_virt_part;
    }
    return *this;
  }

  dim3 grid;
  dim3 threads;
  size_t shared_size;

  unsigned int num_blocks;
  unsigned int num_virt_part;
};


#endif //  KERNELPLAN_H
