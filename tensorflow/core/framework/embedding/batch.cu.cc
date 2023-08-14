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

#include "tensorflow/core/framework/embedding/batch.h"
#include "tensorflow/core/framework/register_types.h"

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
__global__ void InitEmptyCache(char *cache, int key_size, int header_size, int alloc_size, int value_len, int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < limit){
    char *base_ptr = cache + alloc_size * i;
    K *key_ptr = reinterpret_cast<K *>(base_ptr);
    int *freq_ptr = reinterpret_cast<int *>(base_ptr + key_size);
    V *value_ptr = reinterpret_cast<V *>(base_ptr + header_size);
    *key_ptr = static_cast<K>(-1);
    *freq_ptr = 0;
    
    for (int j = 0; j < value_len; j++){
      value_ptr[j] = static_cast<V>(0);
    }
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T1, T2) \
   template __global__ void InitEmptyCache<T1, T2>(char *, int, int, int, int, int);

#define REGISTER_KERNELS_ALL_TYPES(T2) \
   REGISTER_KERNELS_ALL_INDEX(int32, T2) \
   REGISTER_KERNELS_ALL_INDEX(int64, T2)


TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_TYPES)
TF_CALL_int32(REGISTER_KERNELS_ALL_TYPES)
TF_CALL_int64(REGISTER_KERNELS_ALL_TYPES)


#undef REGISTER_KERNELS_ALL_TYPES
#undef REGISTER_KERNELS_ALL_INDEX

template<class K, class V>
__global__ void DeviceInitEmbedding(int *locks, K *keys, char *cache, int key_size, int header_size, int alloc_size, int value_len, int ways, int cache_num, int limit){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < cache_num){
    atomicExch(&locks[i], 0);
  }
  if(i < limit){
    K key = keys[i];
    int cache_id = key % cache_num;
    int possible_place = cache_id * ways;
    bool blocked = true;
    while(blocked) {
      if(0 == atomicCAS(&locks[cache_id], 0, 1)) {
        __threadfence();
        for(int j = possible_place; j < possible_place + ways; j++){
          char *base_ptr = cache + alloc_size * j;
          K *key_ptr = reinterpret_cast<K *>(base_ptr);
          int *freq_ptr = reinterpret_cast<int *>(base_ptr + key_size);
          V *value_ptr = reinterpret_cast<V *>(base_ptr + header_size);
          if(*key_ptr == -1){
            *key_ptr = key;
            for(int k = 0; k < value_len; k++){
              value_ptr[k] = static_cast<V>(key);
            }
            *freq_ptr = 0;
            break;
          }
        }
        __threadfence();
        atomicExch(&locks[cache_id], 0);
        blocked = false;
      }
    }
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T1, T2) \
   template __global__ void DeviceInitEmbedding<T1, T2>(int *, T1 *, char *, int, int, int, int, int, int, int);

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
    K key = keys[i];
    int cache_id = key % cache_num;
    int possible_place = cache_id * ways;
    gather_status[i] = key;
    for(j = possible_place; j < possible_place + ways; j++){
      char *base_ptr = cache + alloc_size * j;
      K *key_ptr = reinterpret_cast<K *>(base_ptr);
      int *freq_ptr = reinterpret_cast<int *>(base_ptr + key_size);
      V *value_ptr = reinterpret_cast<V *>(base_ptr + header_size);

      if(*key_ptr == key){
        gather_status[i] = 0;
        for(int k = 0; k < value_len; k++){
          output[i * value_len + k] = value_ptr[k];
        }
        atomicAdd(freq_ptr, 1);
        break;
      }
      
    }
    if(j == possible_place + ways){
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
__global__ void GatherMissingEmbedding(int *locks, K* keys, char *cache, V *output, int *miss_index, V* memcpy_buffer, int key_size, int header_size, int alloc_size, int value_len, int ways, int cache_num, int miss_count){
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i < miss_count){
    int index = miss_index[i];
    K key = keys[index];
    int cache_id = key % cache_num;
    int possible_place = cache_id * ways;
    
    //Write to output
    for(int j = 0; j < value_len; j++){
      output[index * value_len + j] = memcpy_buffer[i * value_len + j];
    }

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
   template __global__ void GatherMissingEmbedding<T1, T2>(int *, T1 *, char *, T2 *, int *, T2 *, int, int, int, int, int, int, int);

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


}  // namespace tensorflow
#endif  // GOOGLE_CUDA
