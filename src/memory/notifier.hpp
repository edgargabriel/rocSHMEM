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

#ifndef LIBRARY_SRC_MEMORY_NOTIFIER_HPP_
#define LIBRARY_SRC_MEMORY_NOTIFIER_HPP_

#include "../device_proxy.hpp"
#include "../util.hpp"
#include "../atomic.hpp"

namespace rocshmem {

template<detail::atomic::rocshmem_memory_scope Scope>
class Notifier {
};

template<detail::atomic::rocshmem_memory_scope Scope>
class Notifier<detail::atomic::memory_scope_workgroup> {
 public:
  __device__ uint64_t read() { return value_; }

  __device__ void write(uint64_t val) {
    if (is_thread_zero_in_block()) {
      value_ = val;
    }
    publish();
  }

  __device__ void done() { __syncthreads(); }

 private:
  __device__ void publish() {
    if (is_thread_zero_in_block()) {
      __threadfence();
    }
    __syncthreads();
  }

  uint64_t value_{};
};

template <typename ALLOCATOR, detail::atomic::rocshmem_memory_scope Scope>
class NotifierProxy {
  using ProxyT = DeviceProxy<ALLOCATOR, Notifier<Scope>, 1>;

 public:
  __host__ __device__ Notifier* get() { return proxy_.get(); }

 private:
  ProxyT proxy_{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_NOTIFIER_HPP_
