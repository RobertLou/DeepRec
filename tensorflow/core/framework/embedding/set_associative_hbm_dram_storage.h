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

/*     for(int i = 0; i < 256; i++){
      ValuePtr<V> *tmp_value_ptr;
      dram_->Get(static_cast<K>(i), &tmp_value_ptr);
      V* tmp_value;
      tmp_value = tmp_value_ptr->GetValue(0, 0);
      LOG(INFO) << tmp_value[i % 32];
    } */
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
    LOG(INFO) << value_len;
    if(miss_count > 0){
      V *memcpy_buffer_cpu, *memcpy_buffer_gpu;
      memcpy_buffer_cpu = new V[miss_count * value_len];
      cudaMalloc((void**)&memcpy_buffer_gpu, miss_count * value_len * sizeof(V));

      int *missing_index_cpu, *missing_index_gpu;
      missing_index_cpu = new int[miss_count];
      cudaMalloc((void**)&missing_index_gpu, miss_count * sizeof(int));

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

      cudaMemcpy(memcpy_buffer_gpu, memcpy_buffer_cpu,
        miss_count * value_len * sizeof(V), cudaMemcpyHostToDevice);
      cudaMemcpy(missing_index_gpu, missing_index_cpu,
        miss_count * sizeof(int), cudaMemcpyHostToDevice);
      
      hbm_->BatchGetMissing(
        ctx, keys, output, value_len, miss_count, missing_index_gpu, memcpy_buffer_gpu);


      delete []memcpy_buffer_cpu;
      cudaFree(memcpy_buffer_gpu);
      delete []missing_index_cpu;
      cudaFree(missing_index_gpu);
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
  void BatchGetValuePtrs(
      const EmbeddingVarContext<GPUDevice>& ctx,
      const K* keys,
      ValuePtr<V>** value_ptr_list,
      int64 num_of_keys,
      std::vector<std::list<int64>>& copyback_cursor_list,
      std::vector<std::list<int64>>* not_found_cursor_list = nullptr) {
    int num_worker_threads = ctx.worker_threads->num_threads;
    IntraThreadCopyIdAllocator thread_copy_id_alloc(num_worker_threads);
    uint64 main_thread_id = Env::Default()->GetCurrentThreadId();

    std::function<void(std::vector<std::list<int64>>*,
                       int64, int)> set_not_found_list = 0;
    if (not_found_cursor_list != nullptr) {
      set_not_found_list =
          [](std::vector<std::list<int64>>* not_found_cursor_list,
             int64 i, int copy_id) {
        (*not_found_cursor_list)[copy_id].emplace_back(i);
      };
    } else {
      set_not_found_list =
          [](std::vector<std::list<int64>>* not_found_cursor_list,
             int64 i, int copy_id) {};
    }

    auto do_work = [this, keys, value_ptr_list, &thread_copy_id_alloc,
                    main_thread_id, &copyback_cursor_list,
                    set_not_found_list, &not_found_cursor_list]
        (int64 start, int64 limit) {
      int copy_id =
          thread_copy_id_alloc.GetCopyIdOfThread(main_thread_id);
      for (int64 i = start; i < limit; i++) {
        Status s = Get(keys[i], &value_ptr_list[i]);
        if (s.ok()) {
          int64 copyback_flag =
              (int64)value_ptr_list[i] >> copyback_flag_offset_bits_;
          RemoveCopyBackFlagInValuePtr(&value_ptr_list[i]);
          if (copyback_flag == CopyBackFlag::COPYBACK) {
            copyback_cursor_list[copy_id].emplace_back(i);
          }
        } else {
          value_ptr_list[i] = nullptr;
          set_not_found_list(not_found_cursor_list, i, copy_id);
        }
      }
    };
    auto worker_threads = ctx.worker_threads;
    Shard(worker_threads->num_threads,
          worker_threads->workers, num_of_keys,
          1000, do_work);

    for (int i = 1; i < worker_threads->num_threads + 1; i++) {
      if (copyback_cursor_list[i].size()>0) {
        copyback_cursor_list[0].splice(copyback_cursor_list[0].end(),
                                       copyback_cursor_list[i]);
      }
    }

    if (not_found_cursor_list != nullptr) {
      for (int i = 1; i < worker_threads->num_threads + 1; i++) {
        if ((*not_found_cursor_list)[i].size()>0) {
          (*not_found_cursor_list)[0].splice(
              (*not_found_cursor_list)[0].end(),
              (*not_found_cursor_list)[i]);
        }
      }
    }
  }

  void CopyEmbeddingsFromDramToHbm(const EmbeddingVarContext<GPUDevice>& ctx,
                                   const K* keys,
                                   ValuePtr<V>** value_ptr_list,
                                   std::list<int64>& copyback_cursors,
                                   int64 value_len) {
    int64 total = copyback_cursors.size();
    std::vector<ValuePtr<V>*> gpu_value_ptrs(total);
    std::vector<K> copyback_keys(total);
    std::vector<int64> memory_index(total);
    //Create Hbm ValuePtrs.
    {
      int64 i = 0;
      auto it = copyback_cursors.cbegin();
      //Mutex with eviction thread
      mutex_lock l(memory_pool_mu_);
      for ( ; it != copyback_cursors.cend(); ++it, ++i) {
        int64 j = *it;
        memory_index[i] = j;
        ValuePtr<V>* gpu_value_ptr = hbm_->CreateValuePtr(value_len);
        V* val_ptr = embedding_mem_pool_->Allocate();
        bool flag = gpu_value_ptr->SetPtr(val_ptr);
        if (!flag) {
          embedding_mem_pool_->Deallocate(val_ptr);
        }
        memcpy((char *)gpu_value_ptr->GetPtr(),
               (char *)value_ptr_list[j]->GetPtr(),
               sizeof(FixedLengthHeader));
        gpu_value_ptrs[i] = gpu_value_ptr;
        copyback_keys[i] = keys[*it];
      }
    }
    MultiTierStorage<K, V>::CopyEmbeddingsFromDramToHbm(
        ctx, keys, value_ptr_list, copyback_cursors,
        memory_index, gpu_value_ptrs, value_len);

    //Insert copyback ids to hbm hash table.
    auto do_insert = [this, copyback_keys, gpu_value_ptrs,
                      memory_index, value_ptr_list]
        (int64 start, int64 limit) {
      for (int64 i = start; i < limit; i++) {
        Status s = hbm_->TryInsert(
            copyback_keys[i], gpu_value_ptrs[i]);
        if (!s.ok()) {
          {
            mutex_lock l(memory_pool_mu_);
            embedding_mem_pool_->Deallocate(
                gpu_value_ptrs[i]->GetValue(0, 0));
          }
          delete gpu_value_ptrs[i];
          hbm_->Get(copyback_keys[i], &value_ptr_list[memory_index[i]]);
        }
      }
    };
    auto worker_threads = ctx.worker_threads;
    Shard(worker_threads->num_threads, worker_threads->workers,
          total, 100000, do_insert);
  }

  void CreateValuePtrs(const EmbeddingVarContext<GPUDevice>& ctx,
                       const K* keys,
                       ValuePtr<V>** value_ptr_list,
                       std::list<int64>& not_found_cursors,
                       int64 value_len) {
    int64 total = not_found_cursors.size();
    if (total > 0) {
      std::vector<std::pair<int64, ValuePtr<V>*>> insert_pairs(total);
      std::vector<int64> cursor_index(total);
      //Create Hbm ValuePtrs.
      {
        int64 i = 0;
        auto it = not_found_cursors.cbegin();
        //Mutex with eviction thread
        mutex_lock l(memory_pool_mu_);
        for ( ; it != not_found_cursors.cend(); ++it, ++i) {
          int64 j = *it;
          cursor_index[i] = j;
          ValuePtr<V>* gpu_value_ptr = hbm_->CreateValuePtr(value_len);
          V* val_ptr = embedding_mem_pool_->Allocate();
          bool flag = gpu_value_ptr->SetPtr(val_ptr);
          if (!flag) {
            embedding_mem_pool_->Deallocate(val_ptr);
          }
          value_ptr_list[j] = gpu_value_ptr;
          insert_pairs[i].first = keys[j];
          insert_pairs[i].second = value_ptr_list[j];
        }
      }

      //Insert copyback ids to hbm hash table.
      auto do_insert = [this, insert_pairs, value_ptr_list, cursor_index]
          (int64 start, int64 limit) {
        for (int64 i = start; i < limit; i++) {
          Status s = hbm_->TryInsert(
              insert_pairs[i].first, insert_pairs[i].second);
          if (!s.ok()) {
            {
              mutex_lock l(memory_pool_mu_);
              embedding_mem_pool_->Deallocate(
                  insert_pairs[i].second->GetValue(0, 0));
            }
            delete insert_pairs[i].second;
            hbm_->Get(insert_pairs[i].first, &value_ptr_list[cursor_index[i]]);
          }
        }
      };
      auto worker_threads = ctx.worker_threads;
      Shard(worker_threads->num_threads, worker_threads->workers,
            total, 100000, do_insert);
    }
  }

  void AddCopyBackFlagToValuePtr(
      ValuePtr<V>** value_ptr, CopyBackFlag copyback_flag) {
    int64 tmp = ((int64)copyback_flag) << copyback_flag_offset_bits_;
    tmp = ((int64)*value_ptr) | tmp;
    *value_ptr = reinterpret_cast<ValuePtr<V>*>(tmp);
  }

  void RemoveCopyBackFlagInValuePtr(ValuePtr<V>** value_ptr) {
    int64 tmp = (1L << (copyback_flag_offset_bits_)) - 1;
    tmp = ((int64)*value_ptr) & tmp;
    *value_ptr = reinterpret_cast<ValuePtr<V>*>(tmp);
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
