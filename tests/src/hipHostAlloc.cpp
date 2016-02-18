#include <assert.h>
#include <stdio.h>
#include "hip_runtime.h"

#include "test_common.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))


#define WIDTH     1024
#define HEIGHT    1024

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  16
#define THREADS_PER_BLOCK_Z  1

__global__ void
vectoradd_float(hipLaunchParm lp,
                float *a, const float *b, const float *c,
                int width, int height) {

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    int i = y * width + x;
    if ( i < (width * height)) {
      a[i] = b[i] + c[i];
    }
}


int main() {

  float* hostA;
  float* hostB;
  float* hostC;

  float* deviceA;
  float* deviceB;
  float* deviceC;

  int i;
  int errors;

  HIPCHECK(hipHostAlloc((void **)&hostA, NUM * sizeof(float), hipHostAllocMapped));
  HIPCHECK(hipHostAlloc((void **)&hostB, NUM * sizeof(float), hipHostAllocMapped));
  HIPCHECK(hipHostAlloc((void **)&hostC, NUM * sizeof(float), hipHostAllocMapped));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = (float)i;
    hostC[i] = (float)i*100.0f;
  }

  HIPCHECK(hipHostGetDevicePointer((void **)&deviceA, (void **)&hostA, 0));
  HIPCHECK(hipHostGetDevicePointer((void **)&deviceB, (void **)&hostB, 0));
  HIPCHECK(hipHostGetDevicePointer((void **)&deviceC, (void **)&hostC, 0));

  hipLaunchKernel(vectoradd_float,
                  dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  deviceA ,deviceB ,deviceC ,WIDTH ,HEIGHT);

  HIPCHECK(hipDeviceSynchronize());

  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != (hostB[i] + hostC[i])) {
      errors++;
    }
  }

  if (errors!=0) {
    failed("FAILED: %d errors\n",errors);
  } else {
      passed();
  }

  hipFreeHost(hostA);
  hipFreeHost(hostB);
  hipFreeHost(hostC);


  return errors;
}
