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

#include "roc_shmem/roc_shmem.hpp"
#include "../context_incl.hpp"
#include "context_ipc_tmpl_device.hpp"
#include "../util.hpp"
#include "ipc_team.hpp"

namespace rocshmem {

__device__ void IPCContext::internal_direct_barrier(int pe, int PE_start,
                                                      int stride, int n_pes,
                                                      int64_t *pSync) {
  int64_t flag_val = 1;
  if (pe == PE_start) {
    // Go through all PE offsets (except current offset = 0)
    // and wait until they all reach
#if defined(__gfx90a__)
    __threadfence_system();
#endif /* __gfx90a__ */
    for (size_t i = 1; i < n_pes; i++) {
      wait_until(&pSync[i], ROC_SHMEM_CMP_EQ, flag_val);
      pSync[i] = ROC_SHMEM_SYNC_VALUE;
    }
    threadfence_system();

    // Announce to other PEs that all have reached
    for (size_t i = 1, j = PE_start + stride; i < n_pes; ++i, j += stride) {
      internal_putmem(&pSync[0], &flag_val, sizeof(*pSync), j);
#if defined(__gfx90a__)
        __threadfence_system();
#endif /* __gfx90a__ */
    }
  } else {
    // Mark current PE offset as reached
    size_t pe_offset = (pe - PE_start) / stride;
    internal_putmem(&pSync[pe_offset], &flag_val, sizeof(*pSync), PE_start);
#if defined(__gfx90a__)
    __threadfence_system();
#endif /* __gfx90a__ */
    wait_until(&pSync[0], ROC_SHMEM_CMP_EQ, flag_val);
    pSync[0] = ROC_SHMEM_SYNC_VALUE;
    threadfence_system();
  }
}

__device__ void IPCContext::internal_atomic_barrier(int pe, int PE_start,
                                                      int stride, int n_pes,
                                                      int64_t *pSync) {
  int64_t flag_val = 1;
  if (pe == PE_start) {
    wait_until(&pSync[0], ROC_SHMEM_CMP_EQ, (int64_t)(n_pes - 1));
    pSync[0] = ROC_SHMEM_SYNC_VALUE;
    threadfence_system();

    for (size_t i = 1, j = PE_start + stride; i < n_pes; ++i, j += stride) {
      internal_putmem(&pSync[0], &flag_val, sizeof(*pSync), j);
    }
  } else {
    amo_add<int64_t>(&pSync[0], flag_val, PE_start);
    wait_until(&pSync[0], ROC_SHMEM_CMP_EQ, flag_val);
    pSync[0] = ROC_SHMEM_SYNC_VALUE;
    threadfence_system();
  }
}

// Uses PE values that are relative to world
__device__ void IPCContext::internal_sync(int pe, int PE_start, int stride,
                                          int PE_size, int64_t *pSync) {
#ifdef USE_COOPERATIVE_GROUPS
  cg::grid_group grid = cg::this_grid();
  grid.sync();
  if (0 == grid.thread_rank())
#else
  notifier_->sync();
  if (0 == get_flat_id())
#endif /* USE_COOPERATIVE_GROUPS */
  {
    if (PE_size < 64) {
      internal_direct_barrier(pe, PE_start, stride, PE_size, pSync);
    } else {
      internal_atomic_barrier(pe, PE_start, stride, PE_size, pSync);
    }
  }
  __threadfence();
#ifdef USE_COOPERATIVE_GROUPS
  grid.sync();
#else
  notifier_->sync();
#endif /* USE_COOPERATIVE_GROUPS */
}

__device__ void IPCContext::sync(roc_shmem_team_t team) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  /**
   * Ensure that the stride is a multiple of 2.
   */
  int log_pe_stride = static_cast<int>(team_obj->tinfo_wrt_world->log_stride);
  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = (1 << log_pe_stride);
  int pe_size = team_obj->num_pes;

  internal_sync(pe, pe_start, pe_stride, pe_size, barrier_sync);
}

__device__ void IPCContext::sync_all() {
  internal_sync(my_pe, 0, 1, num_pes, barrier_sync);
}

__device__ void IPCContext::barrier_all() {
  if (is_thread_zero_in_block()) {
    quiet();
  }
  sync_all();
  __syncthreads();
}

}  // namespace rocshmem
