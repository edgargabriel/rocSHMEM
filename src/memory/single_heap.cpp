/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "single_heap.hpp"

#include <sstream>
#include "util.hpp"

#if defined USE_ALLOC_DLMALLOC
#include "dlmalloc.hpp"
#elif defined USE_ALLOC_POW2BINS
#include "address_record.hpp"
#include "pow2_bins.hpp"
#else
#error "You need to have one of USE_ALLOC_DLMALLOC, USE_ALLOC_POW2BINS set to ON"
#endif

#include "hip_allocator.hpp"

namespace rocshmem {

SingleHeap::SingleHeap() {

  int hip_dev_id{};
  CHECK_HIP(hipGetDevice(&hip_dev_id));
  printf("Got device_id %d\n", hip_dev_id);
  std::string heap_mem_type = envvar::heap_mem_type;

  if (heap_mem_type.empty()) {
    // Note: not using get_arch_name(hip_dev_id) from ../util.cpp because 
    // the required data structure are not being initialized in the unit tests.
    char arch_name[256];
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, hip_dev_id));
    std::snprintf(arch_name, sizeof(arch_name), "%s",prop.gcnArchName);

    if (strncmp(arch_name, "gfx1201", strlen("gfx1201")) == 0) {
      heap_mem_ = new HeapMemoryType<HIPAllocatorFinegrained>(envvar::heap_size.get_value());
    } else {
#if defined HIP_SUPPORTS_MALLOC_UNCACHED
      heap_mem_ = new HeapMemoryType<HIPAllocatorUncached>(envvar::heap_size.get_value());
#else
      heap_mem_ = new HeapMemoryType<HIPAllocatorFinegrained>(envvar::heap_size.get_value());
#endif
    }
  } else {
    if (heap_mem_type.compare("coarsegrained") == 0) {
      heap_mem_ = new HeapMemoryType<HIPAllocatorCoarsegrained>(envvar::heap_size.get_value());
    }
    else if (heap_mem_type.compare("finegrained") == 0) {
      heap_mem_ = new HeapMemoryType<HIPAllocatorFinegrained>(envvar::heap_size.get_value());
    }
    else if (heap_mem_type.compare("uncached") == 0) {
#if defined HIP_SUPPORTS_MALLOC_UNCACHED
      heap_mem_ = new HeapMemoryType<HIPAllocatorUncached>(envvar::heap_size.get_value());
#else
      printf("Uncached Heap memory type requested, but ROCm version does not support Uncached memory. Aborting.\n");
      abort();
#endif
    }
  }
  assert(heap_mem_ != nullptr);

#if defined USE_ALLOC_DLMALLOC
  if (heap_mem_->type_ == AllocatorTypeCoarsegrained) {
    strat_ = new DLAllocatorStrategy<HeapMemoryType<HIPAllocatorCoarsegrained>>(reinterpret_cast<HeapMemoryType<HIPAllocatorCoarsegrained> *>(heap_mem_));
  } else if (heap_mem_->type_ == AllocatorTypeFinegrained){
    strat_ = new DLAllocatorStrategy<HeapMemoryType<HIPAllocatorFinegrained>>(reinterpret_cast<HeapMemoryType<HIPAllocatorFinegrained> *>(heap_mem_));
  } else if (heap_mem_->type_ == AllocatorTypeUncached){
    strat_ = new DLAllocatorStrategy<HeapMemoryType<HIPAllocatorUncached>>(reinterpret_cast<HeapMemoryType<HIPAllocatorUncached> *>(heap_mem_));
  }

#elif defined USE_ALLOC_POW2BINS
  /**
   * @brief Helper type for address records
   */
  using AR_T = AddressRecord;
  /**
   * @brief Helper type for allocation strategy
   */
 strat_ = new Pow2Bins<AR_T, *heap_mem_>();
#endif // defined USE_ALLOC_POW2BINS
}

void SingleHeap::malloc(void** ptr, size_t size) {
  strat_->alloc(reinterpret_cast<char**>(ptr), size);
}

__device__ void SingleHeap::malloc(void** ptr, size_t size) {}

void SingleHeap::free(void* ptr) {
  if (!ptr) {
    return;
  }
  strat_->free(reinterpret_cast<char*>(ptr));
}

__device__ void SingleHeap::free(void* ptr) {}

void* SingleHeap::realloc(void* ptr, size_t size) { return nullptr; }

void* SingleHeap::malign(size_t alignment, size_t size) { return nullptr; }

char* SingleHeap::get_base_ptr() { return heap_mem_->get_ptr(); }

size_t SingleHeap::get_size() { return heap_mem_->get_size(); }

size_t SingleHeap::get_used() { return strat_->get_used(); }

size_t SingleHeap::get_avail() { return get_size() - get_used(); }

}  // namespace rocshmem
