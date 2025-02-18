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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_RO_NET_GPU_TEMPLATES_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_RO_NET_GPU_TEMPLATES_HPP_

#include "rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "commands_types.hpp"
#include "context_ro_device.hpp"
#include "queue_proxy.hpp"
#include "ro_net_team.hpp"

namespace rocshmem {

template <typename T>
struct GetROType {};

template <>
struct GetROType<char> {
  static constexpr ro_net_types Type{RO_NET_CHAR};
};

template <>
struct GetROType<unsigned char> {
  static constexpr ro_net_types Type{RO_NET_CHAR};
};

template <>
struct GetROType<signed char> {
  static constexpr ro_net_types Type{RO_NET_CHAR};
};

template <>
struct GetROType<unsigned short> {
  static constexpr ro_net_types Type{RO_NET_SHORT};
};

template <>
struct GetROType<unsigned int> {
  static constexpr ro_net_types Type{RO_NET_INT};
};

template <>
struct GetROType<unsigned long> {
  static constexpr ro_net_types Type{RO_NET_LONG};
};

template <>
struct GetROType<unsigned long long> {
  static constexpr ro_net_types Type{RO_NET_LONG_LONG};
};

template <>
struct GetROType<float> {
  static constexpr ro_net_types Type{RO_NET_FLOAT};
};

template <>
struct GetROType<double> {
  static constexpr ro_net_types Type{RO_NET_DOUBLE};
};

template <>
struct GetROType<int> {
  static constexpr ro_net_types Type{RO_NET_INT};
};

template <>
struct GetROType<short> {
  static constexpr ro_net_types Type{RO_NET_SHORT};
};

template <>
struct GetROType<long> {
  static constexpr ro_net_types Type{RO_NET_LONG};
};

template <>
struct GetROType<long long> {
  static constexpr ro_net_types Type{RO_NET_LONG_LONG};
};

template <>
struct GetROType<long double> {
  static constexpr ro_net_types Type{RO_NET_LONG_DOUBLE};
};

/******************************************************************************
 ********************************* DEVICE API *********************************
 *****************************************************************************/

template <typename T, ROCSHMEM_OP Op>
__device__ void ROContext::to_all(rocshmem_team_t team, T *dest,
                                  const T *source, int nreduce) {
  if (!is_thread_zero_in_block()) {
    __syncthreads();
    return;
  }

  ROTeam *team_obj{reinterpret_cast<ROTeam *>(team)};

  build_queue_element(RO_NET_TEAM_TO_ALL, dest, const_cast<T *>(source),
                      nreduce, 0, 0, 0, 0, nullptr, nullptr, team_obj->mpi_comm,
                      ro_net_win_id, block_handle, true, Op, GetROType<T>::Type);

  __syncthreads();
}

template <typename T, ROCSHMEM_OP Op>
__device__ void ROContext::to_all(T *dest, const T *source, int nreduce,
                                  int PE_start, int logPE_stride, int PE_size,
                                  T *pWrk, long *pSync) {
  if (!is_thread_zero_in_block()) {
    __syncthreads();
    return;
  }

  build_queue_element(RO_NET_TO_ALL, dest, const_cast<T *>(source), nreduce,
                      PE_start, logPE_stride, PE_size, 0, pWrk, pSync,
                      (MPI_Comm)NULL, ro_net_win_id, block_handle, true, Op,
                      GetROType<T>::Type);

  __syncthreads();
}

template <typename T>
__device__ void ROContext::put(T *dest, const T *source, size_t nelems,
                               int pe) {
  size_t size{sizeof(T) * nelems};
  putmem(dest, source, size, pe);
}

template <typename T>
__device__ void ROContext::put_nbi(T *dest, const T *source, size_t nelems,
                                   int pe) {
  size_t size{sizeof(T) * nelems};
  putmem_nbi(const_cast<T *>(dest), const_cast<T *>(source), size, pe);
}

template <typename T>
__device__ void ROContext::p(T *dest, T value, int pe) {
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    long L_offset{reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe]};
    ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[pe] + L_offset,
                     reinterpret_cast<void *>(&value), sizeof(T));
  } else {
    build_queue_element(RO_NET_P, dest, &value, sizeof(T), pe, 0, 0, 0, nullptr,
                        nullptr, (MPI_Comm)NULL, ro_net_win_id,
                        block_handle, true);
  }
}

template <typename T>
__device__ T ROContext::g(const T *source, int pe) {
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    const char *src_typed{reinterpret_cast<const char *>(source)};
    long L_offset{const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[my_pe]};
    T dest;
    ipcImpl_.ipcCopy(&dest, ipcImpl_.ipc_bases[pe] + L_offset, sizeof(T));
    return dest;
  } else {
    int thread_id{get_flat_block_id()};
    int block_size{get_flat_block_size()};
    int offset{get_flat_grid_id() * block_size + thread_id};

    char *base_dest{block_handle->g_ret};
    char *dest{&base_dest[offset * sizeof(int64_t)]};
    get<T>(reinterpret_cast<T *>(dest), source, 1, pe);
    return *(reinterpret_cast<T *>(dest));
  }
}

template <typename T>
__device__ void ROContext::get(T *dest, const T *source, size_t nelems,
                               int pe) {
  size_t size{sizeof(T) * nelems};
  getmem(dest, source, size, pe);
}

template <typename T>
__device__ void ROContext::get_nbi(T *dest, const T *source, size_t nelems,
                                   int pe) {
  size_t size{sizeof(T) * nelems};
  getmem_nbi(dest, source, size, pe);
}

template <typename T>
__device__ T ROContext::amo_fetch_cas(void *dst, T value, T cond, int pe) {
  auto source{get_unused_atomic()};
  build_queue_element(RO_NET_AMO_FCAS, dst, reinterpret_cast<T *>(source),
                      value, pe, 0, 0, 0,
                      reinterpret_cast<void *>(static_cast<long long>(cond)),
                      nullptr, (MPI_Comm)NULL, ro_net_win_id, block_handle, true,
                      ROCSHMEM_SUM, GetROType<T>::Type);
  __threadfence();
  return *source;
}

template <typename T>
__device__ void ROContext::amo_cas(void *dst, T value, T cond, int pe) {
  T ret{amo_fetch_cas(dst, value, cond, pe)};
}

template <typename T>
__device__ T ROContext::amo_fetch_add(void *dst, T value, int pe) {
  auto source{get_unused_atomic()};
  build_queue_element(RO_NET_AMO_FOP, dst, reinterpret_cast<T *>(source), value,
                      pe, 0, 0, 0, nullptr, nullptr, (MPI_Comm)NULL,
                      ro_net_win_id, block_handle, true, ROCSHMEM_SUM,
                      GetROType<T>::Type);
  __threadfence();
  return *source;
}

template <typename T>
__device__ void ROContext::amo_add(void *dst, T value, int pe) {
  T ret{amo_fetch_add(dst, value, pe)};
}

template <typename T>
__device__ T ROContext::amo_swap(void *dst, T value, int pe) {
  auto source{get_unused_atomic()};
  build_queue_element(RO_NET_AMO_FOP, dst, reinterpret_cast<void *>(source),
                      value, pe, 0, 0, 0, nullptr, nullptr, (MPI_Comm)NULL,
                      ro_net_win_id, block_handle, true, ROCSHMEM_REPLACE,
                      GetROType<T>::Type);
  __threadfence();
  return *source;
}

template <typename T>
__device__ void ROContext::amo_set(void *dst, T value, int pe) {
  T ret{amo_swap(dst, value, pe)};
}

template <typename T>
__device__ T ROContext::amo_fetch_and(void *dst, T value, int pe) {
  auto source{get_unused_atomic()};
  build_queue_element(RO_NET_AMO_FOP, dst, reinterpret_cast<void *>(source),
                      value, pe, 0, 0, 0, nullptr, nullptr, (MPI_Comm)NULL,
                      ro_net_win_id, block_handle, true, ROCSHMEM_AND,
                      GetROType<T>::Type);
  __threadfence();
  return *source;
}

template <typename T>
__device__ void ROContext::amo_and(void *dst, T value, int pe) {
  T ret{amo_fetch_and(dst, value, pe)};
}

template <typename T>
__device__ T ROContext::amo_fetch_or(void *dst, T value, int pe) {
  auto source{get_unused_atomic()};
  build_queue_element(RO_NET_AMO_FOP, dst, reinterpret_cast<void *>(source),
                      value, pe, 0, 0, 0, nullptr, nullptr, (MPI_Comm)NULL,
                      ro_net_win_id, block_handle, true, ROCSHMEM_OR,
                      GetROType<T>::Type);
  __threadfence();
  return *source;
}

template <typename T>
__device__ void ROContext::amo_or(void *dst, T value, int pe) {
  T ret{amo_fetch_or(dst, value, pe)};
}

template <typename T>
__device__ T ROContext::amo_fetch_xor(void *dst, T value, int pe) {
  auto source{get_unused_atomic()};
  build_queue_element(RO_NET_AMO_FOP, dst, reinterpret_cast<void *>(source),
                      value, pe, 0, 0, 0, nullptr, nullptr, (MPI_Comm)NULL,
                      ro_net_win_id, block_handle, true, ROCSHMEM_XOR,
                      GetROType<T>::Type);
  __threadfence();
  return *source;
}

template <typename T>
__device__ void ROContext::amo_xor(void *dst, T value, int pe) {
  T ret{amo_fetch_xor(dst, value, pe)};
}

template <typename T>
__device__ void ROContext::broadcast(rocshmem_team_t team, T *dest,
                                     const T *source, int nelems, int pe_root) {
  if (!is_thread_zero_in_block()) {
    __syncthreads();
    return;
  }

  ROTeam *team_obj{reinterpret_cast<ROTeam *>(team)};

  build_queue_element(RO_NET_TEAM_BROADCAST, dest, const_cast<T *>(source),
                      nelems, 0, 0, 0, pe_root, nullptr, nullptr,
                      team_obj->mpi_comm, ro_net_win_id, block_handle, true,
                      ROCSHMEM_SUM, GetROType<T>::Type);

  __syncthreads();
}

template <typename T>
__device__ void ROContext::broadcast(T *dest, const T *source, int nelems,
                                     int pe_root, int pe_start,
                                     int log_pe_stride, int pe_size,
                                     long *p_sync) {
  if (!is_thread_zero_in_block()) {
    __syncthreads();
    return;
  }

  build_queue_element(RO_NET_BROADCAST, dest, const_cast<T *>(source), nelems,
                      pe_start, log_pe_stride, pe_size, pe_root, nullptr,
                      p_sync, (MPI_Comm)NULL, ro_net_win_id, block_handle, true,
                      ROCSHMEM_SUM, GetROType<T>::Type);

  __syncthreads();
}

template <typename T>
__device__ void ROContext::alltoall(rocshmem_team_t team, T *dest,
                                    const T *source, int nelems) {
  if (!is_thread_zero_in_block()) {
    __syncthreads();
    return;
  }

  ROTeam *team_obj{reinterpret_cast<ROTeam *>(team)};

  build_queue_element(RO_NET_ALLTOALL, dest, const_cast<T *>(source), nelems, 0,
                      0, 0, 0, team_obj->ata_buffer, nullptr,
                      team_obj->mpi_comm, ro_net_win_id, block_handle, true,
                      ROCSHMEM_SUM, GetROType<T>::Type);

  __syncthreads();
}

template <typename T>
__device__ void ROContext::fcollect(rocshmem_team_t team, T *dest,
                                    const T *source, int nelems) {
  if (!is_thread_zero_in_block()) {
    __syncthreads();
    return;
  }

  ROTeam *team_obj{reinterpret_cast<ROTeam *>(team)};

  build_queue_element(RO_NET_FCOLLECT, dest, const_cast<T *>(source), nelems, 0,
                      0, 0, 0, team_obj->ata_buffer, nullptr,
                      team_obj->mpi_comm, ro_net_win_id, block_handle, true,
                      ROCSHMEM_SUM, GetROType<T>::Type);

  __syncthreads();
}

/**
 * WG and WAVE level API
 */

template <typename T>
__device__ void ROContext::put_wg(T *dest, const T *source, size_t nelems,
                                  int pe) {
  size_t size{sizeof(T) * nelems};
  putmem_wg(dest, source, size, pe);
}

template <typename T>
__device__ void ROContext::put_nbi_wg(T *dest, const T *source, size_t nelems,
                                      int pe) {
  size_t size{sizeof(T) * nelems};
  putmem_nbi_wg(dest, source, size, pe);
}

template <typename T>
__device__ void ROContext::put_wave(T *dest, const T *source, size_t nelems,
                                    int pe) {
  size_t size{sizeof(T) * nelems};
  putmem_wave(dest, source, size, pe);
}

template <typename T>
__device__ void ROContext::put_nbi_wave(T *dest, const T *source, size_t nelems,
                                        int pe) {
  size_t size{sizeof(T) * nelems};
  putmem_nbi_wave(dest, source, size, pe);
}

template <typename T>
__device__ void ROContext::get_wg(T *dest, const T *source, size_t nelems,
                                  int pe) {
  size_t size{sizeof(T) * nelems};
  getmem_wg(dest, source, size, pe);
}

template <typename T>
__device__ void ROContext::get_nbi_wg(T *dest, const T *source, size_t nelems,
                                      int pe) {
  size_t size{sizeof(T) * nelems};
  getmem_nbi_wg(dest, source, size, pe);
}

template <typename T>
__device__ void ROContext::get_wave(T *dest, const T *source, size_t nelems,
                                    int pe) {
  size_t size{sizeof(T) * nelems};
  getmem_wave(dest, source, size, pe);
}

template <typename T>
__device__ void ROContext::get_nbi_wave(T *dest, const T *source, size_t nelems,
                                        int pe) {
  size_t size{sizeof(T) * nelems};
  getmem_nbi_wave(dest, source, size, pe);
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_RO_NET_GPU_TEMPLATES_HPP_
