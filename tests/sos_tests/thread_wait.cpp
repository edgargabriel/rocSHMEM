/*
 *  Copyright (c) 2017 Intel Corporation. All rights reserved.
 *  This software is available to you under the BSD license below:
 *
 *      Redistribution and use in source and binary forms, with or
 *      without modification, are permitted provided that the following
 *      conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* Thread wait test: Test whether a store performed by one thead will wake up a
 * second thread from a call to rocshmem_wait. */

#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

static long *shr_var;

static void *src_thread_fn(void *arg) {
  /* Try to get the dst thread to enter wait before the call to sleep */
  sleep(1);

  /* This should wake up the waiting dst thread */
  *shr_var = 1;

  /* Quiet should provide a store fence */
  rocshmem_quiet();
  return NULL;
}

static void *dst_thread_fn(void *arg) {
  rocshmem_long_wait_until(shr_var, ROCSHMEM_CMP_NE, 0);
  printf("shr_var is now %ld\n", *shr_var);
  return NULL;
}

int main(int argc, char *argv[]) {
  int tl, ret;
  pthread_t src_thread, dst_thread;

  rocshmem_init_thread(ROCSHMEM_THREAD_MULTIPLE, &tl);

  if (tl != ROCSHMEM_THREAD_MULTIPLE) {
    printf("Init failed (requested thread level %d, got %d)\n",
           ROCSHMEM_THREAD_MULTIPLE, tl);
    rocshmem_global_exit(1);
  }

  shr_var = (long *)rocshmem_malloc(sizeof(long));
  *shr_var = 0;

  pthread_create(&dst_thread, NULL, &dst_thread_fn, NULL);
  pthread_create(&src_thread, NULL, &src_thread_fn, NULL);

  pthread_join(dst_thread, NULL);
  pthread_join(src_thread, NULL);

  rocshmem_free(shr_var);

  rocshmem_finalize();

  return 0;
}
