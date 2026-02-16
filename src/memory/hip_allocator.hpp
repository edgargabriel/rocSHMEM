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

#ifndef LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_HPP_
#define LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_HPP_

/**
 * @file hip_allocator.hpp
 *
 * @brief Contains HIP wrapper class for memory allocator
 */

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "memory_allocator.hpp"

#include <hip/hip_runtime_api.h>

#include <cstdlib>
#include <limits>
#include <vector>

// `hipDeviceMallocUncached` was introduced at ROCm 5.5
#if (HIP_VERSION_MAJOR > 5) || \
    (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 5)
#define HIP_SUPPORTS_MALLOC_UNCACHED
#elif defined USE_HEAP_DEVICE_UNCACHED
#error "USE_HEAP_DEVICE_UNCACHED unsupported in this HIP version"
#endif
namespace rocshmem {

enum HIPIpcHandleType {
  HandleTypeLegacy = 0,
  HandleTypePosix,
  HandleTypeFabric,
  HandleTypeLast
};

enum HIPAllocatorType {
  AllocatorTypeCoarsegrained = 0,
  AllocatorTypeFinegrained,
  AllocatorTypeUncached,
  AllocatorTypeVMM,
  AllocatorTypeLast
};

class HIPIpcHandleVec {
public:
  virtual HIPIpcHandleType GetIpcHandleType() = 0;
  virtual void* GetHandleVecElem(int elem) = 0;

};

class HIPIpcHandleLegacyVec : public HIPIpcHandleVec {
public:
  friend class HIPAllocator;

  HIPIpcHandleType GetIpcHandleType() { return HandleTypeLegacy; }

  void* GetHandleVecElem(int elem)
  {
    return reinterpret_cast<void*> (&this->handle[elem]);
  }

protected:
  std::vector<hipIpcMemHandle_t> handle;

};

class HIPAllocator : public MemoryAllocator {
 public:

  HIPAllocator() : MemoryAllocator(hipMalloc, hipFree) {}

  HIPAllocator(hipError_t (*hip_alloc_fn)(void**, size_t),
               hipError_t (*hip_free_fn)(void*)) :
      MemoryAllocator (hip_alloc_fn, hip_free_fn) {}

  HIPAllocator (hipError_t (*hip_alloc_fn)(void**, size_t, unsigned),
                hipError_t (*hip_free_fn)(void*), unsigned flags) :
    MemoryAllocator (hip_alloc_fn, hip_free_fn, flags) {}

  HIPAllocatorType type = AllocatorTypeLast;

  hipError_t GetIpcHandle(void *dev_ptr, void *handle)
  {
    return hipIpcGetMemHandle(reinterpret_cast<hipIpcMemHandle_t *>(handle), dev_ptr);
  }

  hipError_t OpenIpcHandle(void **dev_ptr, void *handle)
  {
    return hipIpcOpenMemHandle(dev_ptr, *(reinterpret_cast<hipIpcMemHandle_t *>(handle)),
                               hipIpcMemLazyEnablePeerAccess);
  }

  hipError_t CloseIpcHandle(void *dev_ptr)
  {
    return hipIpcCloseMemHandle(dev_ptr);
  }

  size_t GetIpcHandleSize()
  {
    return sizeof(hipIpcMemHandle_t);
  }

  HIPIpcHandleVec* AllocateIpcHandleVec(int num_elems)
  {
    HIPIpcHandleLegacyVec* vec = new HIPIpcHandleLegacyVec();
    vec->handle.resize(num_elems);
    return vec;
  }
};

class HIPAllocatorCoarsegrained : public HIPAllocator {
 public:
  HIPAllocatorCoarsegrained()
      : HIPAllocator(hipMalloc, hipFree) {
    type = AllocatorTypeCoarsegrained;
  }
};

class HIPAllocatorFinegrained : public HIPAllocator {
 public:
  HIPAllocatorFinegrained()
      : HIPAllocator(hipExtMallocWithFlags, hipFree,
                     hipDeviceMallocFinegrained) {
    type = AllocatorTypeFinegrained;
  }
};

#if defined HIP_SUPPORTS_MALLOC_UNCACHED
class HIPAllocatorUncached : public HIPAllocator {
 public:
  HIPAllocatorUncached()
      : HIPAllocator(hipExtMallocWithFlags, hipFree,
                     hipDeviceMallocUncached) {
    type = AllocatorTypeUncached;
  }
};

// The default fine-grained coherence allocator is the uncached allocator
using HIPDefaultFinegrainedAllocator = HIPAllocatorUncached;
#else
// The default fine-grained coherence allocator is the fine-grained allocator
using HIPDefaultFinegrainedAllocator = HIPAllocatorFinegrained;
#endif

class HIPHostAllocator : public MemoryAllocator {
 public:
  HIPHostAllocator()
      : MemoryAllocator(hipHostMalloc, hipFree, hipHostMallocCoherent) {}
};

template <class T>
class StdAllocatorHIP {
 public:
  typedef T value_type;

  StdAllocatorHIP()
  {
    allocator_ = new HIPDefaultFinegrainedAllocator();
  }

  template <class U>
  constexpr StdAllocatorHIP(const StdAllocatorHIP<U>&) noexcept
  {
    allocator_ = new HIPDefaultFinegrainedAllocator();
  }

  [[nodiscard]] T* allocate(size_t n) {
    if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }

    T* p{nullptr};
    allocator_->allocate(reinterpret_cast<void**>(&p), n * sizeof(T));
    if (p) {
      return p;
    }

    throw std::bad_alloc();
  }

  void deallocate(T* p, [[maybe_unused]] size_t n) noexcept {
    allocator_->deallocate(p);
  }

 private:
  HIPAllocator *allocator_{nullptr};
};

template <class T, class U>
bool operator==(const StdAllocatorHIP<T>&, const StdAllocatorHIP<U>&) {
  return true;
}

template <class T, class U>
bool operator!=(const StdAllocatorHIP<T>&, const StdAllocatorHIP<U>&) {
  return false;
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_HPP_
