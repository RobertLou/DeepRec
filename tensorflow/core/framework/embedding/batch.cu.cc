/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/embedding/batch.h"

namespace tensorflow {

template<class V>
__global__ void BatchInit(V** val, V** default_value, int value_len, int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < limit * value_len){
    *(val[i / value_len] + i % value_len) = *(default_value[i / value_len] + i % value_len);
  }
}

template __global__ void BatchInit<int>(int**, int**, int, int);
template __global__ void BatchInit<float>(float**, float**, int, int);
template __global__ void BatchInit<double>(double**, double**, int, int);
template __global__ void BatchInit<long long>(long long**, long long**, int, int);

template<class V>
__global__ void BatchInitOneDefault(V** val, V* default_value, int value_len, int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < limit * value_len){
    *(val[i / value_len] + i % value_len) = default_value[i % value_len];
  }
}

template __global__ void BatchInitOneDefault<int>(int**, int*, int, int);
template __global__ void BatchInitOneDefault<float>(float**, float*, int, int);
template __global__ void BatchInitOneDefault<double>(double**, double*, int, int);
template __global__ void BatchInitOneDefault<long long>(long long**, long long*, int, int);

template<class V>
__global__ void BatchCopy(V** batch, V* val_base, int value_len, int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < limit * value_len){  
    val_base[i] = *(batch[i / value_len] + i % value_len);
  }
}

/*
__global__ void BatchCopy(V** batch, V* val_base, int value_len, int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < limit){
      for(int j = 0;j < value_len; ++j){
      val_base[i * value_len + j] = *(batch[i] + j);
    }
  }
}
*/

template __global__ void BatchCopy<int>(int**, int*, int, int);
template __global__ void BatchCopy<float>(float**, float*, int, int);
template __global__ void BatchCopy<double>(double**, double*, int, int);
template __global__ void BatchCopy<long long>(long long**, long long*, int, int);

template<class V>
__global__ void SparseApplyAdagradGPU(V** a, V** v, V* g, float lr, int embedding_dim, int limit, bool* init_flags, V* default_value) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / embedding_dim;
  int item_pos = i % embedding_dim;

  if(i < limit * embedding_dim){
    if(init_flags[item_id]){
      *(a[item_id] + item_pos) = default_value[item_pos];
    }
    *(a[item_id] + item_pos) += g[i] * g[i];
    *(v[item_id] + item_pos) -= lr * g[i] * rsqrt(*(a[item_id] + item_pos));
  }
}

/*
__global__ void SparseApplyAdagradGPU(V** a, V** v, V* g, float lr, int embedding_dim, int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < limit){
    for(int j = 0;j < embedding_dim; ++j) {
      *(a[i] + j) += g[i * embedding_dim + j] * g[i * embedding_dim + j];
      *(v[i] + j) -= lr * g[i * embedding_dim + j] / sqrt(*(a[i] + j));
    }
  }
}
*/

//template __global__ void SparseApplyAdagradGPU<int>(int**, int**, int*, float, int, int);
template __global__ void SparseApplyAdagradGPU<float>(float**, float**, float*, float, int, int, bool*, float*);
template __global__ void SparseApplyAdagradGPU<double>(double**, double**, double*, float, int, int, bool*, double*);
//template __global__ void SparseApplyAdagradGPU<long long>(long long**, long long**, long long*, float, int, int);


template<class V>
__global__ void CopyEmbedding(V** batch, V* batch_data_space, int total_dims_, int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < limit){
    for(int j = 0;j < total_dims_;j++){
      batch_data_space[i * total_dims_ + j] = *(batch[i] + j);
    }
  }
}

template __global__ void CopyEmbedding<int>(int**, int*, int, int);
template __global__ void CopyEmbedding<float>(float**, float*, int, int);
template __global__ void CopyEmbedding<double>(double**, double*, int, int);
template __global__ void CopyEmbedding<long long>(long long**, long long*, int, int);

}  // namespace tensorflow

