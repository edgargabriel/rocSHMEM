/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef _TESTER_HPP_
#define _TESTER_HPP_

#include <rocshmem/rocshmem.hpp>
#include <vector>

#include "tester_arguments.hpp"

/******************************************************************************
 * TESTER CLASS TYPES
 *****************************************************************************/
enum TestType {
  GetTestType = 0,
  GetNBITestType = 1,
  PutTestType = 2,
  PutNBITestType = 3,
  GetSwarmTestType = 4,
  // ReductionTestType = 5,
  AMO_FAddTestType = 6,
  AMO_FIncTestType = 7,
  AMO_FetchTestType = 8,
  AMO_FCswapTestType = 9,
  AMO_AddTestType = 10,
  AMO_IncTestType = 11,
  AMO_CswapTestType = 12,
  InitTestType = 13,
  PingPongTestType = 14,
  //    BarrierTestType         = 15, // Deprecated in OpenSHMEM 1.5
  RandomAccessTestType = 16,
  BarrierAllTestType = 17,
  SyncAllTestType = 18,
  SyncTestType = 19,
  // BroadcastTestType = 20,
  CollectTestType = 21,
  FCollectTestType = 22,
  AllToAllTestType = 23,
  AllToAllsTestType = 24,
  ShmemPtrTestType = 25,
  PTestType = 26,
  GTestType = 27,
  WGGetTestType = 28,
  WGGetNBITestType = 29,
  WGPutTestType = 30,
  WGPutNBITestType = 31,
  WAVEGetTestType = 32,
  WAVEGetNBITestType = 33,
  WAVEPutTestType = 34,
  WAVEPutNBITestType = 35,
  TeamBroadcastTestType = 36,
  TeamReductionTestType = 37,
  TeamCtxGetTestType = 38,
  TeamCtxGetNBITestType = 39,
  TeamCtxPutTestType = 40,
  TeamCtxPutNBITestType = 41,
  TeamCtxInfraTestType = 42,
  PutNBIMRTestType = 43,
  AMO_SetTestType = 44,
  AMO_SwapTestType = 45,
  AMO_FetchAndTestType = 46,
  AMO_FetchOrTestType = 47,
  AMO_FetchXorTestType = 48,
  AMO_AndTestType = 49,
  AMO_OrTestType = 50,
  AMO_XorTestType = 51,
  PingAllTestType = 52,
  PutSignalTestType = 53,
  WGPutSignalTestType = 54,
  WAVEPutSignalTestType = 55,
  PutSignalNBITestType = 56,
  WGPutSignalNBITestType = 57,
  WAVEPutSignalNBITestType = 58,
  SignalFetchTestType = 59,
  WGSignalFetchTestType = 60,
  WAVESignalFetchTestType = 61,
};

enum OpType { PutType = 0, GetType = 1 };

typedef int ShmemContextType;

/******************************************************************************
 * TESTER INTERFACE
 *****************************************************************************/
class Tester {
 public:
  explicit Tester(TesterArguments args);
  virtual ~Tester();

  void execute();

  static std::vector<Tester *> create(TesterArguments args);

 protected:
  virtual void resetBuffers(uint64_t size) = 0;

  virtual void preLaunchKernel() {}

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            uint64_t size) = 0;

  virtual void postLaunchKernel() {}

  virtual void verifyResults(uint64_t size) = 0;

  int num_msgs = 0;
  int num_timed_msgs = 0;
  int num_warps = 0;
  int bw_factor = 1;
  int device_id = 0;

  TesterArguments args;

  TestType _type;
  ShmemContextType _shmem_context = 8;  // SHMEM_CTX_WP_PRIVATE

  hipStream_t stream;
  hipDeviceProp_t deviceProps;

  uint64_t *timer = nullptr;

 private:
  bool _print_header = 1;
  void print(uint64_t size);

  void barrier();

  uint64_t gpuCyclesToMicroseconds(uint64_t cycles);

  uint64_t timerAvgInMicroseconds();

  bool peLaunchesKernel();

  hipEvent_t start_event;
  hipEvent_t stop_event;
};

#define CHECK_HIP(cmd)                                                        \
  {                                                                           \
    hipError_t error = cmd;                                                   \
    if (error != hipSuccess) {                                                \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), \
              error, __FILE__, __LINE__);                                     \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }

#endif /* _TESTER_HPP */
