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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SET_ASSOCIATIVE_HBM_DRAM_STORAGE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SET_ASSOCIATIVE_HBM_DRAM_STORAGE_H_

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/framework/embedding/lockless_hash_map_cpu.h"
#include "tensorflow/core/framework/embedding/multi_tier_storage.h"
#include "tensorflow/core/framework/embedding/single_tier_storage.h"
#include "tensorflow/core/framework/embedding/hbm_storage_iterator.h"
#include "tensorflow/core/framework/embedding/intra_thread_copy_id_allocator.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {
using se::DeviceMemoryBase;
using se::Stream;

template <class V>
class ValuePtr;

void SyncWithEventMgr(se::Stream* stream, EventMgr* event_mgr);

namespace embedding {
template<typename K, typename V>
class SetAssociativeHbmDramStorage : public MultiTierStorage<K, V> {
 public:
  SetAssociativeHbmDramStorage(const StorageConfig& sc, Allocator* gpu_alloc,
      Allocator* cpu_alloc, LayoutCreator<V>* lc, const std::string& name)
      : gpu_alloc_(gpu_alloc),
        MultiTierStorage<K, V>(sc, name) {
    hbm_ = new SetAssociativeHbmStorage<K, V>(sc, gpu_alloc, lc);
    dram_ = new DramStorage<K, V>(sc, cpu_alloc, lc,
        new LocklessHashMapCPU<K, V>(gpu_alloc));
  }

  ~SetAssociativeHbmDramStorage() override {
    delete hbm_;
    delete dram_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(SetAssociativeHbmDramStorage);

  void InitSetAssociativeHbmDramStorage() override {
    hbm_->Init(gpu_alloc_, Storage<K, V>::alloc_len_);
  
    V* default_value;
    default_value = (V *)malloc(sizeof(V) * Storage<K, V>::alloc_len_);
    for(int i = 0; i < Storage<K, V>::alloc_len_; i++){
      default_value[i] = static_cast<V>(i); 
    }

    for(int i = 0; i < 256; i++){
      ValuePtr<V> *tmp_value_ptr = new NormalContiguousValuePtr<V>(ev_allocator(), Storage<K, V>::alloc_len_);
      tmp_value_ptr->GetOrAllocate(ev_allocator(), Storage<K, V>::alloc_len_, default_value, 0, 0);
      dram_->TryInsert(static_cast<K>(i), tmp_value_ptr);
    }

    free(default_value);
  } 

  Status Get(K key, ValuePtr<V>** value_ptr) override {

  }

  void BatchGet(const EmbeddingVarContext<GPUDevice>& ctx,
                const K* keys,
                V* output,
                int64 num_of_keys,
                int64 value_len) override {
    int miss_count = 0;
    K *gather_status;
    gather_status = new K[num_of_keys];
    hbm_->BatchGet(
      ctx, keys, output, num_of_keys, value_len, miss_count, gather_status);
    LOG(INFO) << miss_count;
    if(miss_count > 0){
      V *memcpy_buffer_cpu, *memcpy_buffer_gpu;
      memcpy_buffer_cpu = new V[miss_count * value_len];
      memcpy_buffer_gpu = (V*) gpu_alloc_->AllocateRaw(
            Allocator::kAllocatorAlignment,
            miss_count * value_len * sizeof(V));

      int *missing_index_cpu, *missing_index_gpu;
      missing_index_cpu = new int[miss_count];
      missing_index_gpu = (int *) gpu_alloc_->AllocateRaw(
            Allocator::kAllocatorAlignment,
            miss_count * sizeof(int));

      int missing_place = 0;
      for(int i = 0; i < num_of_keys; i++){
        if(gather_status[i] != 0){
          missing_index_cpu[missing_place] = i;
          ValuePtr<V> *tmp_value_ptr;
          dram_->Get(gather_status[i], &tmp_value_ptr);
          V* tmp_value;
          tmp_value = tmp_value_ptr->GetValue(0, 0);
          memcpy(memcpy_buffer_cpu + missing_place * value_len, tmp_value, value_len * sizeof(V));
          missing_place++;
        }
      }

      auto compute_stream = ctx.compute_stream;
      auto event_mgr = ctx.event_mgr;

      DeviceMemoryBase memcpy_buffer_gpu_dst_ptr(
          memcpy_buffer_gpu, miss_count * value_len * sizeof(V));
      compute_stream->ThenMemcpy(
          &memcpy_buffer_gpu_dst_ptr, memcpy_buffer_cpu, miss_count * value_len * sizeof(V));
      SyncWithEventMgr(compute_stream, event_mgr);

      DeviceMemoryBase missing_index_gpu_dst_ptr(
          missing_index_gpu, miss_count * sizeof(int));
      compute_stream->ThenMemcpy(
          &missing_index_gpu_dst_ptr, missing_index_cpu, miss_count * sizeof(int));
      SyncWithEventMgr(compute_stream, event_mgr);

      
      hbm_->BatchGetMissing(
        ctx, keys, output, value_len, miss_count, missing_index_gpu, memcpy_buffer_gpu);


      PrintGPUCache();

      delete []memcpy_buffer_cpu;
      gpu_alloc_->DeallocateRaw(memcpy_buffer_gpu);
      delete []missing_index_cpu;
      gpu_alloc_->DeallocateRaw(missing_index_gpu);
    }


    delete []gather_status;
  }

  void Insert(K key, ValuePtr<V>* value_ptr) override {

  }


  void Insert(K key, ValuePtr<V>** value_ptr,
              size_t alloc_len) override {

  }

  void InsertToDram(K key, ValuePtr<V>** value_ptr,
              int64 alloc_len) override {
    dram_->Insert(key, value_ptr, alloc_len);
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size) override {


  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size, CopyBackFlag &need_copyback) override {

    return Status::OK();
  }

void ImportToHbm(
      K* ids, int64 size, int64 value_len, int64 emb_index) override {
    V* memcpy_buffer_cpu = new V[size * value_len];
    V* memcpy_buffer_gpu =
        (V*)gpu_alloc_->AllocateRaw(
            Allocator::kAllocatorAlignment,
            size * value_len * sizeof(V));
    K* keys_gpu =
        (K*)gpu_alloc_->AllocateRaw(
            Allocator::kAllocatorAlignment,
            size * sizeof(K));
    ValuePtr<V>** cpu_value_ptrs = new ValuePtr<V>*[size];
    {
      //Mutex with other Import Ops
      mutex_lock l(memory_pool_mu_);
      for (int64 i = 0; i < size; i++) {
        dram_->Get(ids[i], &cpu_value_ptrs[i]);
      }
    }
    
    for (int64 i = 0; i < size; i++) {
      memcpy(memcpy_buffer_cpu + i * value_len,
          cpu_value_ptrs[i]->GetValue(0, 0), value_len * sizeof(V));
    }

    cudaMemcpy(memcpy_buffer_gpu, memcpy_buffer_cpu,
        size * value_len * sizeof(V), cudaMemcpyHostToDevice);
    cudaMemcpy(keys_gpu, ids,
        size * sizeof(K), cudaMemcpyHostToDevice);
    
    hbm_->Restore(
        keys_gpu, value_len, size, memcpy_buffer_gpu);

    PrintGPUCache();

    delete[] memcpy_buffer_cpu;
    delete[] cpu_value_ptrs;
    gpu_alloc_->DeallocateRaw(keys_gpu);
    gpu_alloc_->DeallocateRaw(memcpy_buffer_gpu);
  }

  void PrintGPUCache(){
    mutex_lock l(*hbm_->get_mutex());
    hbm_->PrintCache();
  }

  void CopyEmbeddingsFromCPUToGPU(
      int total, const K* keys,
      const std::list<int64>& copyback_cursor,
      V** memcpy_address, size_t value_len,
      ValuePtr<V> **gpu_value_ptrs, V* memcpy_buffer_gpu,
      se::Stream* compute_stream,
      EventMgr* event_mgr,
      const DeviceBase::CpuWorkerThreads* worker_threads) override {
  }

  Status Remove(K key) override {

  }

  int64 Size() const override {
    int64 total_size = hbm_->Size();
    total_size += dram_->Size();
    return total_size;
  }

  int64 Size(int level) const override {
    if (level == 0) {
      return hbm_->Size();
    } else if (level == 1) {
      return dram_->Size();
    } else {
      return -1;
    }
  }

  int LookupTier(K key) const override {
    Status s = hbm_->Contains(key);
    if (s.ok())
      return 0;
    s = dram_->Contains(key);
    if (s.ok())
      return 1;
    return -1;
  }

  bool IsUseHbm() override {
    return true;
  }

  bool IsSetAssociativeHbm() override {
    return true;
  }

  bool IsSingleHbm() override {
    return false;
  }

  bool IsUsePersistentStorage() override {
    return false;
  }

  void iterator_mutex_lock() override {
    return;
  }

  void iterator_mutex_unlock() override {
    return;
  }

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<ValuePtr<V>* >* value_ptr_list) override {
    {
      mutex_lock l(*(hbm_->get_mutex()));
      TF_CHECK_OK(hbm_->GetSnapshot(key_list, value_ptr_list));
    }
    {
      mutex_lock l(*(dram_->get_mutex()));
      TF_CHECK_OK(dram_->GetSnapshot(key_list, value_ptr_list));
    }
    return Status::OK();
  }

  int64 GetSnapshot(std::vector<K>* key_list,
      std::vector<V* >* value_list,
      std::vector<int64>* version_list,
      std::vector<int64>* freq_list,
      const EmbeddingConfig& emb_config,
      FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
      embedding::Iterator** it) override {
    std::vector<ValuePtr<V>*> hbm_value_ptr_list, dram_value_ptr_list;
    std::vector<K> temp_hbm_key_list, temp_dram_key_list;
    // Get Snapshot of HBM storage
    {
      mutex_lock l(*(hbm_->get_mutex()));
      TF_CHECK_OK(hbm_->GetSnapshot(&temp_hbm_key_list,
                                    &hbm_value_ptr_list));
    }
    // Get Snapshot of DRAM storage.
    {
      mutex_lock l(*(dram_->get_mutex()));
      TF_CHECK_OK(dram_->GetSnapshot(&temp_dram_key_list,
                                     &dram_value_ptr_list));
    }
    *it = new HbmDramIterator<K, V>(temp_hbm_key_list,
                                    temp_dram_key_list,
                                    hbm_value_ptr_list,
                                    dram_value_ptr_list,
                                    Storage<K, V>::alloc_len_,
                                    gpu_alloc_,
                                    emb_config.emb_index);
    // This return value is not the exact number of IDs
    // because the two tables intersect.
    return temp_hbm_key_list.size() + temp_dram_key_list.size();
  }

  Status Shrink(const ShrinkArgs& shrink_args) override {
    hbm_->Shrink(shrink_args);
    dram_->Shrink(shrink_args);
    return Status::OK();
  }

  void CreateEmbeddingMemoryPool(
      Allocator* alloc,
      int64 value_len,
      int64 block_size) override {

  }

  void AllocateMemoryForNewFeatures(
      const std::vector<ValuePtr<V>*>& value_ptr_list) override {

  }

  void AllocateMemoryForNewFeatures(
     ValuePtr<V>** value_ptr_list,
     int64 num_of_value_ptrs) override {

  }

  void BatchEviction() override {

  }

 protected:
  void SetTotalDims(int64 total_dims) override {
    dram_->SetTotalDims(total_dims);
  }

 private:
  SetAssociativeHbmStorage<K, V>* hbm_ = nullptr;
  DramStorage<K, V>* dram_ = nullptr;
  EmbeddingMemoryPool<V>* embedding_mem_pool_ = nullptr;
  Allocator* gpu_alloc_;
  mutex memory_pool_mu_; //ensure thread safety of embedding_mem_pool_
  const int copyback_flag_offset_bits_ = 60;
};
} // embedding
} // tensorflow

#endif  // GOOGLE_CUDA
#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_STORAGE_H_
