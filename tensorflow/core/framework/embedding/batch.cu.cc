/* Copyright 2022 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
======================================================================*/

#if GOOGLE_CUDA

#include <cooperative_groups.h>

#include "tensorflow/core/framework/embedding/batch.h"
#include "tensorflow/core/framework/register_types.h"

namespace cg = cooperative_groups;

namespace tensorflow {
namespace embedding {
template<class V>
__global__ void BatchCopy(V** batch, V* val_base, int value_len,
    int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / value_len;
  int item_pos = i % value_len;

  if (i < limit * value_len) {
    val_base[i] = *(batch[item_id] + item_pos);
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T) \
   template __global__ void BatchCopy<T>(T**, T*, int, int);
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_int32(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_int64(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX

template<class V>
__global__ void BatchUnpack(V** dev_value_address,
    V* memcpy_buffer_gpu, int value_len, int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / value_len;
  int item_pos = i % value_len;

  if (i < limit * value_len) {
    *(dev_value_address[item_id] + item_pos) = memcpy_buffer_gpu[i];
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T)                       \
  template __global__ void BatchUnpack<T>(T**, T*, int, int);
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_int32(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_int64(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX

template<class V>
__global__ void CopyEmbedding(V** batch, V** batch_data_space,
    int total_dims, int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / total_dims;
  int item_pos = i % total_dims;

  if (i < limit  * total_dims) {
    *(batch_data_space[item_id] + item_pos) = *(batch[item_id] + item_pos);
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T) \
   template __global__ void CopyEmbedding<T>(T**, T**, int, int);
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
}  // namespace embedding

template<class V>
__global__ void SparseApplyAdagradGPU(V** a, V** v, const V* g, V lr,
    int embedding_dim, long long int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / embedding_dim;
  int item_pos = i % embedding_dim;

  if (i < limit * embedding_dim) {
    *(a[item_id] + item_pos) += g[i] * g[i];
    *(v[item_id] + item_pos) -=
        lr * g[i] * rsqrt(*(a[item_id] + item_pos));
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T)                \
  template __global__ void SparseApplyAdagradGPU<T>( \
    T**, T**, const T*, T, int, long long int);
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX

template<class V>
__global__ void SparseApplyAdamGPU(V** var, V** m, V** v,
    const V* g, V alpha, V beta1, V beta2, V epsilon,
    int embedding_dim, long long int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / embedding_dim;
  int item_pos = i % embedding_dim;

  if (i < limit * embedding_dim) {
    *(m[item_id] + item_pos) +=
        (g[i] - (*(m[item_id] + item_pos))) * (1.0 - beta1);
    *(v[item_id] + item_pos) +=
        (g[i] * g[i] - (*(v[item_id] + item_pos))) * (1.0 - beta2);
    *(var[item_id] + item_pos) -=
        (*(m[item_id] + item_pos) * alpha) /
        (sqrt(*(v[item_id] + item_pos)) + epsilon);
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T)                \
  template __global__ void SparseApplyAdamGPU<T>( \
    T**, T**, T**, const T*, T, \
    T, T, T, int, long long int);
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX

template<class V>
__global__ void SparseApplyAdamAsyncGPU(
    V** var, V** m, V** v,
    const V* g, V lr, V beta1, V beta2, V epsilon,
    V* beta1_power_ptr, V* beta2_power_ptr,
    int embedding_dim, long long int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / embedding_dim;
  int item_pos = i % embedding_dim;

  if (i < limit * embedding_dim) {
    V beta1_power = *beta1_power_ptr;
    V beta2_power = *beta2_power_ptr;
    const V alpha = lr *
        sqrt(static_cast<V>(1) - beta2_power) /
        (static_cast<V>(1) - beta1_power);
    *(m[item_id] + item_pos) = *(m[item_id] + item_pos) * beta1 +
        g[i] * (1 - beta1);
    *(v[item_id] + item_pos) = *(v[item_id] + item_pos) * beta2 +
        g[i] * g[i] * (1 - beta2);
    *(var[item_id] + item_pos) -=
        (*(m[item_id] + item_pos) * alpha) /
        (sqrt(*(v[item_id] + item_pos)) + epsilon);
  }
  __syncthreads();

  if (i == 0) {
    *beta1_power_ptr *= beta1;
    *beta2_power_ptr *= beta2;
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T)                \
  template __global__ void SparseApplyAdamAsyncGPU<T>( \
    T**, T**, T**, const T*, T, \
    T, T, T, T*, T*, int, long long int);
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX

template<class V>
__global__ void SparseApplyAdamAsyncSparseRmspropGPU(
    V** var, V** m, V** v,
    const V* g, V lr, V beta1, V beta2, V epsilon,
    int embedding_dim, long long int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / embedding_dim;
  int item_pos = i % embedding_dim;

  if (i < limit * embedding_dim) {
    *(v[item_id] + item_pos) =  *(v[item_id] + item_pos) * beta2 +
        g[i] * g[i] * (1.0 - beta2);
    *(m[item_id] + item_pos) = *(m[item_id] + item_pos) * beta1 +
        rsqrt(*(v[item_id] + item_pos) + epsilon) *
        lr * g[i];
    *(var[item_id] + item_pos) -= *(m[item_id] + item_pos);
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T)                \
  template __global__ void SparseApplyAdamAsyncSparseRmspropGPU<T>( \
    T**, T**, T**, const T*, T, \
    T, T, T, int, long long int);
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX

template<class V>
__global__ void SparseApplyAdamWGPU(V** var, V** m, V** v,
    const V* g, V alpha, V beta1, V beta2, V epsilon,
    V weight_decay, int embedding_dim, long long int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / embedding_dim;
  int item_pos = i % embedding_dim;

  if (i < limit * embedding_dim) {
    *(m[item_id] + item_pos) +=
        (g[i] - *(m[item_id] + item_pos)) * (1.0 - beta1);
    *(v[item_id] + item_pos) +=
        (g[i] * g[i] - *(v[item_id] + item_pos)) * (1.0 - beta2);
    *(var[item_id] + item_pos) -=
        (*(m[item_id] + item_pos) * alpha) /
        (sqrt(*(v[item_id] + item_pos)) + epsilon) +
        weight_decay * (*(var[item_id] + item_pos));
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T)                \
  template __global__ void SparseApplyAdamWGPU<T>( \
    T**, T**, T**, const T*, T, \
    T, T, T, T, int, long long int);
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX

template<class K, class V>
__global__ void InitEmptyCache(int *locks, char *cache, int key_size, int header_size, int alloc_size, int value_len, int cache_num, int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i < cache_num){
    locks[i] = 0;
  }
  if(i < limit){
    char *base_ptr = cache + alloc_size * i;
    K *key_ptr = reinterpret_cast<K *>(base_ptr);
    int *freq_ptr = reinterpret_cast<int *>(base_ptr + key_size);
    V *value_ptr = reinterpret_cast<V *>(base_ptr + header_size);
    *key_ptr = static_cast<K>(-1);
    *freq_ptr = -1;
    
    for (int j = 0; j < value_len; j++){
      value_ptr[j] = static_cast<V>(0);
    }
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T1, T2) \
   template __global__ void InitEmptyCache<T1, T2>(int*, char *, int, int, int, int, int, int);

#define REGISTER_KERNELS_ALL_TYPES(T2) \
   REGISTER_KERNELS_ALL_INDEX(int32, T2) \
   REGISTER_KERNELS_ALL_INDEX(int64, T2)


TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_TYPES)
TF_CALL_int32(REGISTER_KERNELS_ALL_TYPES)
TF_CALL_int64(REGISTER_KERNELS_ALL_TYPES)


#undef REGISTER_KERNELS_ALL_TYPES
#undef REGISTER_KERNELS_ALL_INDEX

template<class K, class V>
__global__ void GatherEmbedding(K* keys, char *cache, V *output, int *miss_count, K *gather_status, int key_size, int header_size, int alloc_size, int value_len, int ways, int cache_num, int limit){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j;
  if (i == 0){
    *miss_count = 0;
  }
  if(i < limit){
    int key_index = i / value_len;
    int embedding_index = i % value_len;
    K key = keys[key_index];
    int cache_id = key % cache_num;
    int possible_place = cache_id * ways;
    if(embedding_index == 0){
      gather_status[key_index] = key;
    }
    for(j = possible_place; j < possible_place + ways; j++){
      char *base_ptr = cache + alloc_size * j;
      K *key_ptr = reinterpret_cast<K *>(base_ptr);
      int *freq_ptr = reinterpret_cast<int *>(base_ptr + key_size);
      V *value_ptr = reinterpret_cast<V *>(base_ptr + header_size);

      if(*key_ptr == key){
        output[i] = value_ptr[embedding_index];

        if(embedding_index == 0){
          gather_status[key_index] = 0;
          atomicAdd(freq_ptr, 1);
        }
        
        break;
      }
      
    }

    if(embedding_index == 0 && j == possible_place + ways){
      atomicAdd(miss_count, 1);
    }
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T1, T2) \
   template __global__ void GatherEmbedding<T1, T2>(T1 *, char *, T2 *, int *, T1 *, int, int, int, int, int, int, int);

#define REGISTER_KERNELS_ALL_TYPES(T2) \
   REGISTER_KERNELS_ALL_INDEX(int32, T2) \
   REGISTER_KERNELS_ALL_INDEX(int64, T2)


TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_TYPES)
TF_CALL_int32(REGISTER_KERNELS_ALL_TYPES)
TF_CALL_int64(REGISTER_KERNELS_ALL_TYPES)


#undef REGISTER_KERNELS_ALL_TYPES
#undef REGISTER_KERNELS_ALL_INDEX

template<class K, class V>
__global__ void GatherMissingEmbedding(int *locks, K* keys, char *cache, V *output, V* default_value_ptr, int *miss_index, 
                                       V* memcpy_buffer, bool *initialize_status, int key_size, int header_size, int alloc_size, 
                                       int value_len, int ways, int cache_num, int miss_count){
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i < miss_count){
    int index = miss_index[i];
    
    //Write to output
    if(initialize_status[i]){
      for(int j = 0; j < value_len; j++){
        output[index * value_len + j] = default_value_ptr[j];
      }
    }
    else{
      for(int j = 0; j < value_len; j++){
        output[index * value_len + j] = memcpy_buffer[i * value_len + j];
      }
    }

    K key = keys[index];
    int cache_id = key % cache_num;
    int possible_place = cache_id * ways;

    //Update Cache
    bool blocked = true;
    int min_freq;
    int min_place;
    while(blocked) {
      if(0 == atomicCAS(&locks[cache_id], 0, 1)) {
        //Find minimum frequency as substitution place
        __threadfence();
        char *base_ptr;
        int *freq_ptr;
        K *key_ptr;
        V *value_ptr;
        base_ptr = cache;
        freq_ptr = reinterpret_cast<int *>(base_ptr + key_size);
        min_freq = *freq_ptr;
        min_place = possible_place;
        for(int j = possible_place; j < possible_place + ways; j++){
          base_ptr = cache + alloc_size * j;
          freq_ptr = reinterpret_cast<int *>(base_ptr + key_size);
          if(*freq_ptr < min_freq){
            min_freq = *freq_ptr;
            min_place = j;
          }
        }

        base_ptr = cache + alloc_size * min_place;
        key_ptr = reinterpret_cast<K *>(base_ptr);
        freq_ptr = reinterpret_cast<int *>(base_ptr + key_size);
        value_ptr = reinterpret_cast<V *>(base_ptr + header_size);

        //substitute
        *key_ptr = key;
        *freq_ptr = 0;
        for(int j = 0; j < value_len; j++){
          value_ptr[j] = memcpy_buffer[i * value_len + j];
        }
        __threadfence();
        atomicExch(&locks[cache_id], 0);
        blocked = false;
      }
    }
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T1, T2) \
   template __global__ void GatherMissingEmbedding<T1, T2>(int *, T1 *, char *, T2 *, T2 *, int *, T2 *, bool*, int, int, int, int, int, int, int);

#define REGISTER_KERNELS_ALL_TYPES(T2) \
   REGISTER_KERNELS_ALL_INDEX(int32, T2) \
   REGISTER_KERNELS_ALL_INDEX(int64, T2)


TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_TYPES)
TF_CALL_int32(REGISTER_KERNELS_ALL_TYPES)
TF_CALL_int64(REGISTER_KERNELS_ALL_TYPES)

#undef REGISTER_KERNELS_ALL_TYPES
#undef REGISTER_KERNELS_ALL_INDEX

template<class K, class V>
__global__ void RestoreEmbedding(int *locks, K* keys, char *cache, V* memcpy_buffer, int key_size, int header_size, int alloc_size, int value_len, int ways, int cache_num, int limit){
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i < limit){
    K key = keys[i];
    int cache_id = key % cache_num;
    int possible_place = cache_id * ways;

    //Update Cache
    bool blocked = true;
    int min_freq;
    int min_place;
    while(blocked) {
      if(0 == atomicCAS(&locks[cache_id], 0, 1)) {
        //Find minimum frequency as substitution place
        __threadfence();
        char *base_ptr;
        int *freq_ptr;
        K *key_ptr;
        V *value_ptr;
        base_ptr = cache;
        freq_ptr = reinterpret_cast<int *>(base_ptr + key_size);
        min_freq = *freq_ptr;
        min_place = possible_place;
        for(int j = possible_place; j < possible_place + ways; j++){
          base_ptr = cache + alloc_size * j;
          freq_ptr = reinterpret_cast<int *>(base_ptr + key_size);
          if(*freq_ptr < min_freq){
            min_freq = *freq_ptr;
            min_place = j;
          }
        }

        base_ptr = cache + alloc_size * min_place;
        key_ptr = reinterpret_cast<K *>(base_ptr);
        freq_ptr = reinterpret_cast<int *>(base_ptr + key_size);
        value_ptr = reinterpret_cast<V *>(base_ptr + header_size);

        //substitute
        *key_ptr = key;
        *freq_ptr = 0;
        for(int j = 0; j < value_len; j++){
          value_ptr[j] = memcpy_buffer[i * value_len + j];
        }
        __threadfence();
        atomicExch(&locks[cache_id], 0);
        blocked = false;
      }
    }
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T1, T2) \
   template __global__ void RestoreEmbedding<T1, T2>(int *, T1 *, char *, T2 *, int, int, int, int, int, int, int);

#define REGISTER_KERNELS_ALL_TYPES(T2) \
   REGISTER_KERNELS_ALL_INDEX(int32, T2) \
   REGISTER_KERNELS_ALL_INDEX(int64, T2)


TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_TYPES)
TF_CALL_int32(REGISTER_KERNELS_ALL_TYPES)
TF_CALL_int64(REGISTER_KERNELS_ALL_TYPES)

#undef REGISTER_KERNELS_ALL_TYPES
#undef REGISTER_KERNELS_ALL_INDEX

template<class V>
__device__ void warp_tile_copy(const int lane_idx,
                                               const int emb_vec_size_in_float, V* d_dst,
                                               const V* d_src) {
#pragma unroll
  for (int i = lane_idx; i < emb_vec_size_in_float; i += WARP_SIZE) {
    d_dst[i] = d_src[i];
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T) \
  template __device__ void warp_tile_copy<T>(const int, const int, T *, const T *);
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_int32(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_int64(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX

// Will be called by multiple thread_block_tile((sub-)warp) on the same mutex
// Expect only one thread_block_tile return to execute critical section at any time
__forceinline__ __device__ void warp_lock_mutex(const cg::thread_block_tile<WARP_SIZE>& warp_tile,
                                                 int& set_mutex) {
  // The first thread of this (sub-)warp to acquire the lock
  if (warp_tile.thread_rank() == 0) {
    while (0 == atomicCAS((int*)&set_mutex, 1, 0))
      ;
  }
  __threadfence();
  warp_tile.sync();  // Synchronize the threads in the (sub-)warp. Execution barrier + memory fence
}

// The (sub-)warp holding the mutex will unlock the mutex after finishing the critical section on a
// set Expect any following (sub-)warp that acquire the mutex can see its modification done in the
// critical section
__forceinline__ __device__ void warp_unlock_mutex(const cg::thread_block_tile<WARP_SIZE>& warp_tile,
                                                   int& set_mutex) {
  __threadfence();
  warp_tile.sync();  // Synchronize the threads in the (sub-)warp. Execution barrier + memory fence
  // The first thread of this (sub-)warp to release the lock
  if (warp_tile.thread_rank() == 0) {
    atomicExch((int*)&set_mutex, 1);
  }
}

// The (sub-)warp doing all reduction to find the slot with min slot_counter
// The slot with min slot_counter is the LR slot.
__forceinline__ __device__ void warp_min_reduction(
    const cg::thread_block_tile<WARP_SIZE>& warp_tile, int& min_slot_counter_val,
    int& slab_distance, int& slot_distance) {
  const int lane_idx = warp_tile.thread_rank();
  slot_distance = lane_idx;

  for (int i = (warp_tile.size() >> 1); i > 0; i = i >> 1) {
    int input_slot_counter_val = warp_tile.shfl_xor(min_slot_counter_val, (int)i);
    int input_slab_distance = warp_tile.shfl_xor(slab_distance, (int)i);
    int input_slot_distance = warp_tile.shfl_xor(slot_distance, (int)i);

    if (input_slot_counter_val == min_slot_counter_val) {
      if (input_slab_distance == slab_distance) {
        if (input_slot_distance < slot_distance) {
          slot_distance = input_slot_distance;
        }
      } else if (input_slab_distance < slab_distance) {
        slab_distance = input_slab_distance;
        slot_distance = input_slot_distance;
      }
    } else if (input_slot_counter_val < min_slot_counter_val) {
      min_slot_counter_val = input_slot_counter_val;
      slab_distance = input_slab_distance;
      slot_distance = input_slot_distance;
    }
  }
}

__global__ void update_kernel_overflow_ignore(int* global_counter,
                                              int* d_missing_len) {
  // Update global counter
  atomicAdd(global_counter, 1);
  *d_missing_len = 0;
}

// Kernel to initialize the GPU cache
// Init every entry of the cache with <unused_key, value> pair
template <class K>
__global__ void init_cache(slab_set<K>* keys, int* slot_counter,
                           int* global_counter, const int num_slot,
                           const K empty_key, int* set_mutex, const int capacity_in_set) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_slot) {
    // Set the key of this slot to unused key
    // Flatten the cache
    K* key_slot = (K*)keys;
    key_slot[idx] = empty_key;

    // Clear the counter for this slot
    slot_counter[idx] = 0;
  }
  // First CUDA thread clear the global counter
  if (idx == 0) {
    global_counter[idx] = 0;
  }

  // First capacity_in_set CUDA thread initialize mutex
  if (idx < capacity_in_set) {
    set_mutex[idx] = 1;
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T) \
  template __global__ void init_cache<T>(slab_set<T> *, int *, int *, const int, const T, int *, const int);
TF_CALL_int32(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_int64(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX


// Kernel to read from cache
// Also update locality information for touched slot
template <class K, class V>
__global__ void get_kernel(const K* d_keys, const int len, V* d_values,
                           const int embedding_vec_size, int* d_missing_index,
                           K* d_missing_keys, int* d_missing_len,
                           int* global_counter,
                           int* slot_counter, const int capacity_in_set,
                           slab_set<K>* keys, V* vals, int* set_mutex,
                           const int task_per_warp_tile) {
  int empty_key = -1;
  // Lane(thread) ID within a warp_tile
  cg::thread_block_tile<WARP_SIZE> warp_tile =
      cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
  const int lane_idx = warp_tile.thread_rank();
  // Warp tile global ID
  const int warp_tile_global_idx =
      (blockIdx.x * (blockDim.x / WARP_SIZE)) + warp_tile.meta_group_rank();
  // The index of key for this thread
  const int key_idx = (warp_tile_global_idx * task_per_warp_tile) + lane_idx;
  // The assigned key for this lane(thread)
  K key;
  // The dst slabset and the dst slab inside this set
  int src_set;
  int src_slab;
  // The variable that contains the missing key
  K missing_key;
  // The variable that contains the index for the missing key
  uint64_t missing_index;
  // The counter for counting the missing key in this warp
  uint8_t warp_missing_counter = 0;
  // Active flag: whether current lane(thread) has unfinished task
  bool active = false;
  if (lane_idx < task_per_warp_tile) {
    if (key_idx < len) {
      active = true;
      key = d_keys[key_idx];
      src_set = key % capacity_in_set;
      src_slab = key % SET_ASSOCIATIVITY;
    }
  }

  // Lane participate in warp_tile ballot to produce warp-level work queue
  unsigned active_mask = warp_tile.ballot(active);

  // The warp-level outer loop: finish all the tasks within the work queue
  while (active_mask != 0) {
    // Next task in the work quere, start from lower index lane(thread)
    int next_lane = __ffs(active_mask) - 1;
    // Broadcast the task and the global index to all lane in the warp_tile
    K next_key = warp_tile.shfl(key, next_lane);
    int next_idx = warp_tile.shfl(key_idx, next_lane);
    int next_set = warp_tile.shfl(src_set, next_lane);
    int next_slab = warp_tile.shfl(src_slab, next_lane);

    // Counter to record how many slab have been searched
    int counter = 0;

    // Working queue before task started
    const unsigned old_active_mask = active_mask;

    // Lock the slabset before operating the slabset
    warp_lock_mutex(warp_tile, set_mutex[next_set]);

    // The warp-level inner loop: finish a single task in the work queue
    while (active_mask == old_active_mask) {
      // When all the slabs inside a slabset have been searched, mark missing task, task is
      // completed
      if (counter >= SET_ASSOCIATIVITY) {
        if (lane_idx == warp_missing_counter) {
          missing_key = next_key;
          missing_index = next_idx;
        }

        if (lane_idx == (int)next_lane) {
          active = false;
        }

        warp_missing_counter++;
        active_mask = warp_tile.ballot(active);
        break;
      }

      // The warp_tile read out the slab
      K read_key = ((K*)(keys[next_set].set_[next_slab].slab_))[lane_idx];

      // Compare the slab data with the target key
      int found_lane = __ffs(warp_tile.ballot(read_key == next_key)) - 1;

      // If found, mark hit task, copy the founded data, the task is completed
      if (found_lane >= 0) {
        int found_offset = (next_set * SET_ASSOCIATIVITY + next_slab) * WARP_SIZE + found_lane;
        if (lane_idx == (int)next_lane) {
          slot_counter[found_offset] = atomicAdd(global_counter, 0);
          active = false;
        }

        warp_tile_copy(lane_idx, embedding_vec_size,
                                  (V*)(d_values + next_idx * embedding_vec_size),
                                  (V*)(vals + found_offset * embedding_vec_size));

        active_mask = warp_tile.ballot(active);
        break;
      }

      // Compare the slab data with empty key, if found empty key, mark missing task, task is
      // completed
      if (warp_tile.ballot(read_key == empty_key) != 0) {
        if (lane_idx == warp_missing_counter) {
          missing_key = next_key;
          missing_index = next_idx;
        }

        if (lane_idx == (int)next_lane) {
          active = false;
        }

        warp_missing_counter++;
        active_mask = warp_tile.ballot(active);
        break;
      }

      // Not found in this slab, the task is not completed, goto searching next slab
      counter++;
      next_slab = (next_slab + 1) % SET_ASSOCIATIVITY;
    }

    // Unlock the slabset after operating the slabset
    warp_unlock_mutex(warp_tile, set_mutex[next_set]);
  }

  // After warp_tile complete the working queue, save the result for output
  // First thread of the warp_tile accumulate the missing length to global variable
  int warp_position;
  if (lane_idx == 0) {
    warp_position = atomicAdd(d_missing_len, (int)warp_missing_counter);
  }
  warp_position = warp_tile.shfl(warp_position, 0);

  if (lane_idx < warp_missing_counter) {
    d_missing_keys[warp_position + lane_idx] = missing_key;
    d_missing_index[warp_position + lane_idx] = missing_index;
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T1, T2) \
   template __global__ void get_kernel<T1, T2>(const T1 *, const int,T2 *, const int, int *, \
      T1 *, int *, int *,int *, const int, slab_set<T1> *, T2 *, int *, const int);                           

#define REGISTER_KERNELS_ALL_TYPES(T2) \
   REGISTER_KERNELS_ALL_INDEX(int32, T2) \
   REGISTER_KERNELS_ALL_INDEX(int64, T2)

TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_TYPES)
TF_CALL_int32(REGISTER_KERNELS_ALL_TYPES)
TF_CALL_int64(REGISTER_KERNELS_ALL_TYPES)

#undef REGISTER_KERNELS_ALL_TYPES
#undef REGISTER_KERNELS_ALL_INDEX

// Kernel to move missing embeddings to output
template<class V>
__global__ void CopyMissingToOutput(V* output, V* memcpy_buffer_gpu, int *missing_index, int value_len, int miss_count) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / value_len;
  int item_pos = i % value_len;

  if (item_id < miss_count) {
    output[missing_index[item_id]] = memcpy_buffer_gpu[item_id * value_len + item_pos];
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T) \
   template __global__ void CopyMissingToOutput<T>(T *, T *, int *, int, int);
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_int32(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_int64(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX


// Kernel to insert or replace the <k,v> pairs into the cache
template <class K, class V>
__global__ void insert_replace_kernel(const K* d_keys, const V* d_values,
                                      const int embedding_vec_size, const int len,
                                      slab_set<K>* keys,  V* vals,
                                      int* slot_counter,
                                      int* set_mutex, int* global_counter,
                                      const int capacity_in_set,
                                      const int task_per_warp_tile) {
  int empty_key = -1;
  // Lane(thread) ID within a warp_tile
  cg::thread_block_tile<WARP_SIZE> warp_tile =
      cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
  const int lane_idx = warp_tile.thread_rank();
  // Warp tile global ID
  const int warp_tile_global_idx =
      (blockIdx.x * (blockDim.x / WARP_SIZE)) + warp_tile.meta_group_rank();
  // The index of key for this thread
  const int key_idx = (warp_tile_global_idx * task_per_warp_tile) + lane_idx;
  // The assigned key for this lane(thread)
  K key;
  // The dst slabset and the dst slab inside this set
  int src_set;
  int src_slab;
  // Active flag: whether current lane(thread) has unfinished task
  bool active = false;
  if (lane_idx < task_per_warp_tile) {
    if (key_idx < len) {
      active = true;
      key = d_keys[key_idx];
      src_set = key % capacity_in_set;
      src_slab = key % SET_ASSOCIATIVITY;
    }
  }

  // Lane participate in warp_tile ballot to produce warp-level work queue
  unsigned active_mask = warp_tile.ballot(active);

  // The warp-level outer loop: finish all the tasks within the work queue
  while (active_mask != 0) {
    // Next task in the work quere, start from lower index lane(thread)
    int next_lane = __ffs(active_mask) - 1;
    // Broadcast the task, the global index and the src slabset and slab to all lane in a warp_tile
    K next_key = warp_tile.shfl(key, next_lane);
    int next_idx = warp_tile.shfl(key_idx, next_lane);
    int next_set = warp_tile.shfl(src_set, next_lane);
    int next_slab = warp_tile.shfl(src_slab, next_lane);
    int first_slab = next_slab;

    // Counter to record how many slab have been searched
    int counter = 0;

    // Variable to keep the min slot counter during the probing
    int max_int = 9999;
    int max_slab_distance = 9999;
    int min_slot_counter_val = max_int;
    // Variable to keep the slab distance for slot with min counter
    int slab_distance = max_slab_distance;
    // Variable to keep the slot distance for slot with min counter within the slab
    int slot_distance;
    // Working queue before task started
    const unsigned old_active_mask = active_mask;

    // Lock the slabset before operating the slabset
    warp_lock_mutex(warp_tile, set_mutex[next_set]);

    // The warp-level inner loop: finish a single task in the work queue
    while (active_mask == old_active_mask) {
      // When all the slabs inside a slabset have been searched
      // and no empty slots or target slots are found. Replace with LRU
      if (counter >= SET_ASSOCIATIVITY) {
        // (sub)Warp all-reduction, the reduction result store in all threads
        warp_min_reduction(warp_tile, min_slot_counter_val,
                                                        slab_distance, slot_distance);

        // Calculate the position of LR slot
        int target_slab = (first_slab + slab_distance) % SET_ASSOCIATIVITY;
        int slot_index =
            (next_set * SET_ASSOCIATIVITY + target_slab) * WARP_SIZE + slot_distance;

        // Replace the LR slot
        if (lane_idx == (int)next_lane) {
          (( K*)(keys[next_set].set_[target_slab].slab_))[slot_distance] = key;
          slot_counter[slot_index] = atomicAdd(global_counter, 0);
        }

        warp_tile_copy<V>(lane_idx, embedding_vec_size,
                                  ( V*)(vals + slot_index * embedding_vec_size),
                                  ( V*)(d_values + next_idx * embedding_vec_size));

        // Replace complete, mark this task completed
        if (lane_idx == (int)next_lane) {
          active = false;
        }

        active_mask = warp_tile.ballot(active);
        break;
      }

      // The warp_tile read out the slab
      K read_key = (( K*)(keys[next_set].set_[next_slab].slab_))[lane_idx];

      // Compare the slab data with the target key
      int found_lane = __ffs(warp_tile.ballot(read_key == next_key)) - 1;

      // If found target key, the insertion/replace is no longer needed.
      // Refresh the slot, the task is completed
      if (found_lane >= 0) {
        int found_offset = (next_set * SET_ASSOCIATIVITY + next_slab) * WARP_SIZE + found_lane;
        if (lane_idx == (int)next_lane) {
          slot_counter[found_offset] = atomicAdd(global_counter, 0);
          active = false;
        }

        active_mask = warp_tile.ballot(active);
        break;
      }

      // Compare the slab data with empty key.
      // If found empty key, do insertion,the task is complete
      found_lane = __ffs(warp_tile.ballot(read_key == empty_key)) - 1;
      if (found_lane >= 0) {
        int found_offset = (next_set * SET_ASSOCIATIVITY + next_slab) * WARP_SIZE + found_lane;

        if (lane_idx == (int)next_lane) {
          (( K*)(keys[next_set].set_[next_slab].slab_))[found_lane] = key;
          slot_counter[found_offset] = atomicAdd(global_counter, 0);
        }

        warp_tile_copy<V>(lane_idx, embedding_vec_size,
                                  ( V*)(vals + found_offset * embedding_vec_size),
                                  ( V*)(d_values + next_idx * embedding_vec_size));

        if (lane_idx == (int)next_lane) {
          active = false;
        }

        active_mask = warp_tile.ballot(active);
        break;
      }

      // If no target or unused slot found in this slab,
      // Refresh LR info, continue probing
      int read_slot_counter =
          slot_counter[(next_set * SET_ASSOCIATIVITY + next_slab) * WARP_SIZE + lane_idx];
      if (read_slot_counter < min_slot_counter_val) {
        min_slot_counter_val = read_slot_counter;
        slab_distance = counter;
      }

      counter++;
      next_slab = (next_slab + 1) % SET_ASSOCIATIVITY;
    }

    // Unlock the slabset after operating the slabset
    warp_unlock_mutex(warp_tile, set_mutex[next_set]);
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T1, T2) \
   template __global__ void insert_replace_kernel<T1, T2>(const T1 *, const T2 *, const int, const int, \
     slab_set<T1> *,  T2 *,  int *,  int *, int *, const int, const int);

#define REGISTER_KERNELS_ALL_TYPES(T2) \
   REGISTER_KERNELS_ALL_INDEX(int32, T2) \
   REGISTER_KERNELS_ALL_INDEX(int64, T2)

TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_TYPES)
TF_CALL_int32(REGISTER_KERNELS_ALL_TYPES)
TF_CALL_int64(REGISTER_KERNELS_ALL_TYPES)

#undef REGISTER_KERNELS_ALL_TYPES
#undef REGISTER_KERNELS_ALL_INDEX

}  // namespace tensorflow
#endif  // GOOGLE_CUDA
