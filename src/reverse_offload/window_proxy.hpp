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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_WINDOW_PROXY_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_WINDOW_PROXY_HPP_

#include "../device_proxy.hpp"
#include "../memory/window_info.hpp"
#include "mpi_transport.hpp"

namespace rocshmem {

template <typename ALLOCATOR>
class WindowProxy {
 public:
  static constexpr size_t MAX_NUM_WINDOWS{32};

 private:
  using ProxyT = DeviceProxy<ALLOCATOR, WindowInfo *, MAX_NUM_WINDOWS>;

 public:
  /*
   * Placement new the memory which is allocated by proxy_
   */
  WindowProxy(SymmetricHeap *heap, MPI_Comm comm) {
    auto *window_info{proxy_.get()};

    for (size_t i{0}; i < MAX_NUM_WINDOWS; i++) {
      window_info[i] =
          new WindowInfo(comm, heap->get_local_heap_base(), heap->get_size());
    }
  }

  /*
   * Since placement new is called in the constructor, then
   * delete must be called manually.
   */
  ~WindowProxy() {
    auto *window_info{proxy_.get()};

    for (size_t i{0}; i < MAX_NUM_WINDOWS; i++) {
      delete window_info[i];
    }
  }

  /*
   * @brief Provide access to the memory referenced by the proxy
   */
  __host__ __device__ WindowInfo **get() { return proxy_.get(); }

 private:
  /*
   * @brief Memory managed by the lifetime of this object
   */
  ProxyT proxy_{};
};

using WindowProxyT = WindowProxy<HostAllocator>;

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_WINDOW_PROXY_HPP_
