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

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_BATCH_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_BATCH_

#if GOOGLE_CUDA

#include<cuda_runtime.h>

namespace tensorflow {
namespace embedding {

template<class V>
__global__ void BatchCopy(V** batch, V* val_base, int value_len,
    int limit);

template<class V>
__global__ void BatchUnpack(V** dev_value_address, V* memcpy_buffer_gpu,
    int value_len, int limit);

template<class V>
__global__ void CopyEmbedding(V** batch, V** batch_data_space,
    int total_dims, int limit);
} //namespace embedding

template<class V>
__global__ void SparseApplyAdagradGPU(V** a, V** v, const V* g, V lr,
    int embedding_dim, long long int limit);

template<class V>
__global__ void SparseApplyAdamGPU(V** var, V** m, V** v,
    const V* g, V alpha, V beta1, V beta2, V epsilon,
    int embedding_dim, long long int limit);

template<class V>
__global__ void SparseApplyAdamAsyncGPU(
    V** var, V** m, V** v,
    const V* g, V lr, V beta1, V beta2, V epsilon,
    V* beta1_power_ptr, V* beta2_power_ptr,
    int embedding_dim, long long int limit);

template<class V>
__global__ void SparseApplyAdamAsyncSparseRmspropGPU(
    V** var, V** m, V** v, const V* g, V lr,
    V beta1, V beta2, V epsilon,
    int embedding_dim, long long int limit);

template<class V>
__global__ void SparseApplyAdamWGPU(V** var, V** m, V** v,
    const V* g, V alpha, V beta1, V beta2, V epsilon,
    V weight_decay, int embedding_dim, long long int limit);

// slab for static slab list
#define WARP_SIZE 32
#define SET_ASSOCIATIVITY 2

template <class K>
struct static_slab {
  K slab_[WARP_SIZE];
};

// Static slablist(slabset) for GPU Cache
template <class K>
struct slab_set {
  static_slab<K> set_[SET_ASSOCIATIVITY];
};

__global__ void update_kernel_overflow_ignore(int* global_counter, int* d_missing_len);

template<class K>
__global__ void init_cache(slab_set<K> *, int *, int *, const int, const K, int *, const int);

template<class K, class V>
__global__ void insert_replace_kernel(const K *, const V *, const int, const int, slab_set<K> *, \
    V *, int *, int *, int *, const int, const int, float);

template<class K, class V>
__global__ void get_kernel(const K *, const int,V *, const int, int *, \
      K *, int *, int *, int *, int *, const int, slab_set<K> *, V *, int * , const int);       


template<class V>
__global__ void CopyMissingToOutput(V *, V *, int *, int, int);
   

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_BATCH_
