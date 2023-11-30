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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SINGLE_TIER_STORAGE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SINGLE_TIER_STORAGE_H_

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/cpu_hash_map_kv.h"
#include "tensorflow/core/framework/embedding/globalstep_shrink_policy.h"
#if GOOGLE_CUDA
#include "tensorflow/core/framework/embedding/gpu_hash_map_kv.h"
#endif // GOOGLE_CUDA
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/l2weight_shrink_policy.h"
#include "tensorflow/core/framework/embedding/layout_creator.h"
#include "tensorflow/core/framework/embedding/leveldb_kv.h"
#include "tensorflow/core/framework/embedding/ssd_hash_kv.h"
#include "tensorflow/core/framework/embedding/storage_config.h"
#include "tensorflow/core/framework/embedding/storage.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
template <class V>
class ValuePtr;

template <class K, class V>
class EmbeddingVar;

template <class K>
struct SsdRecordDescriptor;

namespace embedding {
template<class K, class V>
class DramSsdHashStorage;

template<class K, class V>
class DramPmemStorage;

template<class K, class V>
class DramLevelDBStore;

#if GOOGLE_CUDA
template<class K, class V>
class HbmDramStorage;

template<class K, class V>
class SetAssociativeHbmDramStorage;

template<class K, class V>
class HbmDramSsdStorage;
#endif

template<typename K, typename V>
class SingleTierStorage : public Storage<K, V> {
 public:
  SingleTierStorage(const StorageConfig& sc, Allocator* alloc,
      KVInterface<K, V>* kv, LayoutCreator<V>* lc)
      : kv_(kv), alloc_(alloc), layout_creator_(lc),
        Storage<K, V>(sc) {
    if (sc.embedding_config.steps_to_live != 0) {
      shrink_policy_ =
          new GlobalStepShrinkPolicy<K, V>(
              sc.embedding_config.steps_to_live,
              alloc_,
              kv_);
    } else if (sc.embedding_config.l2_weight_threshold != -1.0) {
      shrink_policy_ =
          new L2WeightShrinkPolicy<K, V>(
              sc.embedding_config.l2_weight_threshold,
              sc.embedding_config.primary_emb_index,
              Storage<K, V>::GetOffset(
                  sc.embedding_config.primary_emb_index),
              alloc_,
              kv_);
    } else {
      shrink_policy_ = new NonShrinkPolicy<K, V>();
    }
  }
  
  ~SingleTierStorage() override {
    mutex_lock l(Storage<K, V>::mu_);
    std::vector<K> key_list;
    std::vector<ValuePtr<V>*> value_ptr_list;
    kv_->GetSnapshot(&key_list, &value_ptr_list);
    for (auto value_ptr : value_ptr_list) {
      value_ptr->Destroy(alloc_);
      delete value_ptr;
    }
    delete kv_;
    delete shrink_policy_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(SingleTierStorage);

  Status Get(K key, ValuePtr<V>** value_ptr) override {
    return kv_->Lookup(key, value_ptr);
  }

  Status Contains(K key) override {
    return kv_->Contains(key);
  }

  virtual void Insert(K key, ValuePtr<V>** value_ptr,
                      size_t alloc_len, bool to_dram = false) override {
    do {
      *value_ptr = layout_creator_->Create(alloc_, alloc_len);
      Status s = kv_->Insert(key, *value_ptr);
      if (s.ok()) {
        break;
      } else {
        (*value_ptr)->Destroy(alloc_);
        delete *value_ptr;
      }
    } while (!(kv_->Lookup(key, value_ptr)).ok());
  }

  virtual void Insert(K key, ValuePtr<V>* value_ptr) override {
    LOG(FATAL)<<"Unsupport Insert(K, ValuePtr<V>*) in SingleTireStorage.";
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size) override {
    Status s = kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      return s;
    }

    *value_ptr = layout_creator_->Create(alloc_, size);
    s = kv_->Insert(key, *value_ptr);
    if (s.ok()) {
      return s;
    }
    // Insert Failed, key already exist
    (*value_ptr)->Destroy(alloc_);
    delete *value_ptr;
    return kv_->Lookup(key, value_ptr);
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size, CopyBackFlag &need_copyback) override {
    need_copyback = NOT_COPYBACK;
    return GetOrCreate(key, value_ptr, size);
  }
 
  Status Remove(K key) override {
    return kv_->Remove(key);
  }

  int64 Size() const override {
    return kv_->Size();
  }
  
  int64 Size(int level) const override {
    if (level > 0) {
      LOG(FATAL) << "Unsupport level>0 in SingleTierStorage.";
    }
    return kv_->Size();
  }

  int64 CacheSize() const override {
    LOG(FATAL) << "Unsupport cachesize in SingleTierStorage.";
    return 0;
  }

  int LookupTier(K key) const override {
    Status s = kv_->Contains(key);
    return (s.ok()) ? 0 : -1;
  }

  void CopyEmbeddingsFromCPUToGPU(
      int total, const K* keys,
      const std::list<int64>& copyback_cursor,
      V** memcpy_address, size_t value_len,
      ValuePtr<V> **gpu_value_ptrs,
      V* memcpy_buffer_gpu,
      se::Stream* compute_stream,
      EventMgr* event_mgr,
      const DeviceBase::CpuWorkerThreads* worker_threads) override {
    LOG(FATAL) << "Unsupport CopyEmbeddingsFromCPUToGPU in SingleTierStorage.";
  };

  BatchCache<K>* Cache() override {
    LOG(FATAL) << "Unsupport Cache in SingleTierStorage.";
    return nullptr;
  }

  void InitCache(embedding::CacheStrategy cache_strategy) override {
    LOG(FATAL) << "Unsupport InitCache in SingleTierStorage.";
  }

  virtual Status BatchCommit(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) override {
    LOG(FATAL) << "Unsupport BatchCommit in Storage: "
               << typeid(this).name();
    return Status::OK();
  }

  virtual Status Commit(K keys, const ValuePtr<V>* value_ptr) {
     LOG(FATAL) << "Unsupport Commit in Storage: "
                << typeid(this).name();
    return Status::OK();
  }

  Status Eviction(K* evict_ids, int64 evict_size) override {
    LOG(FATAL) << "Unsupport Eviction in SingleTierStorage.";
    return Status::OK();
  }

  void CreateEmbeddingMemoryPool(
      Allocator* alloc,
      int64 value_len,
      int64 block_size) override {
    return;
  }

  void AllocateMemoryForNewFeatures(
      const std::vector<ValuePtr<V>*>& value_ptr_list) override {
    return;
  }

  void AllocateMemoryForNewFeatures(
      ValuePtr<V>** value_ptr_list,
      int64 num_of_value_ptrs) override {
    return;
  }

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<ValuePtr<V>*>* value_ptr_list) override {
    mutex_lock l(Storage<K, V>::mu_);
    return kv_->GetSnapshot(key_list, value_ptr_list);
  }

  Status Save(
      const std::string& tensor_name,
      const std::string& prefix,
      BundleWriter* writer,
      const EmbeddingConfig& emb_config,
      ShrinkArgs& shrink_args,
      int64 value_len,
      V* default_value) override {
    std::vector<ValuePtr<V>*> value_ptr_list;
    std::vector<K> key_list_tmp;
    TF_CHECK_OK(kv_->GetSnapshot(
        &key_list_tmp, &value_ptr_list));

    if (emb_config.is_primary()) {
      Shrink(key_list_tmp, value_ptr_list, shrink_args, value_len);
    }

    TF_CHECK_OK((Storage<K, V>::SaveToCheckpoint(
        tensor_name, writer,
        emb_config,
        value_len, default_value,
        key_list_tmp,
        value_ptr_list)));
    return Status::OK();
  }

  void SetAllocLen(int64 value_len, int slot_num) override {
    while (Storage<K, V>::flag_.test_and_set(std::memory_order_acquire));
    // The start address of every slot should be aligned to 16 bytes,
    // otherwise a coredump will happen in the ApplyOp.
    Storage<K, V>::alloc_len_ = Storage<K, V>::ComputeAllocLen(value_len);

    int64 temp = Storage<K, V>::alloc_len_ * slot_num;
    if (temp > Storage<K, V>::total_dims_) {
      Storage<K, V>::total_dims_ = temp;
      SetTotalDims(Storage<K, V>::total_dims_);
    }
    Storage<K, V>::flag_.clear(std::memory_order_release);
  }

  bool IsMultiLevel() override {
    return false;
  }

  bool IsUseHbm() override {
    return false;
  }

  bool IsSetAssociativeHbm() override {
    return false;
  }

  bool IsSingleHbm() override {
    return false;
  }

  bool IsUsePersistentStorage() override {
    return false;
  }

  void Schedule(std::function<void()> fn) override {
    LOG(FATAL) << "Unsupport Schedule in SingleTierStorage.";
  }

 protected:
  virtual void SetTotalDims(int64 total_dims) = 0;

  virtual ValuePtr<V>* CreateValuePtr(int64 size) {
    return layout_creator_->Create(alloc_, size);
  }

  virtual void DestroyValuePtr(ValuePtr<V>* value_ptr) {
    value_ptr->Destroy(alloc_);
    delete value_ptr;
  }
 protected:
  virtual Status RestoreFeatures(int64 key_num, int bucket_num, int64 partition_id,
                                 int64 partition_num, int64 value_len, bool is_filter,
                                 bool is_incr, const EmbeddingConfig& emb_config, 
                                 const Eigen::GpuDevice* device,
                                 FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
                                 RestoreBuffer& restore_buff) override {
    Status s = filter->Restore(key_num, bucket_num, partition_id,
                               partition_num, value_len, is_filter,
                               false/*to_dram*/, is_incr, restore_buff);
    return s;
  }

  virtual void Shrink(std::vector<K>& key_list,
                      std::vector<ValuePtr<V>*>& value_ptr_list,
                      ShrinkArgs& shrink_args,
                      int64 value_len) {
    mutex_lock l(Storage<K, V>::mu_);
    shrink_args.value_len = value_len;
    shrink_policy_->Shrink(
        key_list,
        value_ptr_list,
        shrink_args);
  }

 protected:
  KVInterface<K, V>* kv_;
  ShrinkPolicy<K, V>* shrink_policy_;
  Allocator* alloc_;
  LayoutCreator<V>* layout_creator_;
};

template<typename K, typename V>
class DramStorage : public SingleTierStorage<K, V> {
 public:
  DramStorage(const StorageConfig& sc, Allocator* alloc,
      LayoutCreator<V>* lc,
      KVInterface<K, V>* kv)
      : SingleTierStorage<K, V>(sc, alloc, kv, lc) {}

  ~DramStorage() override {}

  Status BatchCommit(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) {
    return SingleTierStorage<K, V>::kv_->BatchCommit(keys, value_ptrs);
  }

  Status TryInsert(K key, ValuePtr<V>* value_ptr) {
    return SingleTierStorage<K, V>::kv_->Insert(key, value_ptr);
  }

  Status Commit(K keys, const ValuePtr<V>* value_ptr) override{
    return SingleTierStorage<K, V>::kv_->Commit(keys, value_ptr);
  }
 
  TF_DISALLOW_COPY_AND_ASSIGN(DramStorage);
 public:
  friend class DramSsdHashStorage<K, V>;
  friend class DramPmemStorage<K, V>;
  friend class DramLevelDBStore<K, V>;
#if GOOGLE_CUDA
  friend class HbmDramStorage<K, V>;
  friend class SetAssociativeHbmDramStorage<K, V>;
  friend class HbmDramSsdStorage<K, V>;
#endif
 protected:
  void SetTotalDims(int64 total_dims) override {
    SingleTierStorage<K, V>::kv_->SetTotalDims(total_dims);
  }

  void Shrink(std::vector<K>& key_list,
              std::vector<ValuePtr<V>*>& value_ptr_list,
              ShrinkArgs& shrink_args,
              int64 value_len) override {
    SingleTierStorage<K, V>::Shrink(
        key_list,
        value_ptr_list,
        shrink_args,
        value_len);
  }
};

#if GOOGLE_CUDA
template<typename K, typename V>
class HbmStorage : public SingleTierStorage<K, V> {
 public:
  HbmStorage(const StorageConfig& sc, Allocator* alloc,
      LayoutCreator<V>* lc) : SingleTierStorage<K, V>(
          sc, alloc, new GPUHashMapKV<K, V>(sc.embedding_config, alloc), lc) {
  }
  ~HbmStorage() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(HbmStorage);

  bool IsSingleHbm() override {
    return true;
  }

  void SetValueLen(int64 value_len) override {
    SingleTierStorage<K, V>::kv_->SetValueLen(value_len);
  }

  void BatchLookupOrCreate(const K* key, V* val, V* default_v,
      int32 default_v_num,
      size_t n, const Eigen::GpuDevice& device) override {
    SingleTierStorage<K, V>::kv_->BatchLookupOrCreate(key, val,
                                                      default_v,
                                                      default_v_num,
                                                      n, device);
  }

  void BatchLookupOrCreateKeys(const K* key, int32* item_idxs, size_t n,
      const Eigen::GpuDevice& device) override {
    SingleTierStorage<K, V>::kv_->BatchLookupOrCreateKeys(key, n, item_idxs, device);
  }

  void BatchLookup(const Eigen::GpuDevice& device, const K* keys, V* val,
                   size_t n, const V* default_v) override {
    SingleTierStorage<K, V>::kv_->BatchLookup(device, keys, val, n, default_v);
  }

  Status Save(
      const string& tensor_name,
      const string& prefix,
      BundleWriter* writer,
      const EmbeddingConfig& emb_config,
      ShrinkArgs& shrink_args,
      int64 value_len,
      V* default_value) override {
    std::vector<V*> value_ptr_list;
    std::vector<K> key_list_tmp;
    GPUHashMapKV<K, V>* gpu_kv =
        dynamic_cast<GPUHashMapKV<K, V>*>(SingleTierStorage<K, V>::kv_);
    gpu_kv->GetSnapshot(&key_list_tmp, &value_ptr_list, emb_config);

    TF_CHECK_OK((Storage<K, V>::SaveToCheckpoint(
        tensor_name, writer,
        value_len,
        key_list_tmp,
        value_ptr_list)));

    if (value_ptr_list.size() > 0) {
      TypedAllocator::Deallocate(
          cpu_allocator(), value_ptr_list[0],
          value_ptr_list.size() * value_len);
    }
    return Status::OK();
  }

  GPUHashTable<K, V>* HashTable() override {
    return SingleTierStorage<K, V>::kv_->HashTable();
  }
 protected:
  Status RestoreFeatures(int64 key_num, int bucket_num, int64 partition_id,
                         int64 partition_num, int64 value_len, bool is_filter,
                         bool is_incr, const EmbeddingConfig& emb_config,
                         const Eigen::GpuDevice* device,
                         FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
                         RestoreBuffer& restore_buff) override {
    K* key_buff = (K*)restore_buff.key_buffer;
    V* value_buff = (V*)restore_buff.value_buffer;
    std::vector<K> key_import;
    std::vector<V> value_import;
    for (auto i = 0; i < key_num; ++i) {
      if (*(key_buff + i) % bucket_num % partition_num != partition_id) {
        LOG(INFO) << "skip EV key:" << *(key_buff + i);
        continue;
      }
      key_import.emplace_back(*(key_buff + i));
      auto row_offset = value_buff + i * value_len;
      for (int j = 0; j < value_len; j++) {
        value_import.emplace_back(*(row_offset + j));
      }
    }
    GPUHashMapKV<K, V>* gpu_kv =
        dynamic_cast<GPUHashMapKV<K, V>*>(SingleTierStorage<K, V>::kv_);
    gpu_kv->Import(key_import, value_import, device, emb_config);
    return Status::OK();
  }

  void SetTotalDims(int64 total_dims) override {}
};

template<typename K, typename V>
class HbmStorageWithCpuKv: public SingleTierStorage<K, V> {
 public:
  HbmStorageWithCpuKv(const StorageConfig& sc, Allocator* alloc,
      LayoutCreator<V>* lc) : SingleTierStorage<K, V>(
          sc, alloc, new LocklessHashMap<K, V>(), lc) {
  }

  ~HbmStorageWithCpuKv() override {}

  void Insert(K key, ValuePtr<V>* value_ptr) override {
    do {
      Status s = SingleTierStorage<K, V>::kv_->Insert(key, value_ptr);
      if (s.ok()) {
        break;
      } else {
        value_ptr->Destroy(SingleTierStorage<K, V>::alloc_);
        delete value_ptr;
      }
    } while (!(SingleTierStorage<K, V>::kv_->Lookup(key, &value_ptr)).ok());
  }

  void Insert(K key, ValuePtr<V>** value_ptr,
              size_t alloc_len, bool to_dram = false) override {
    SingleTierStorage<K, V>::Insert(key, value_ptr, alloc_len, to_dram);
  }

  Status TryInsert(K key, ValuePtr<V>* value_ptr) {
    return SingleTierStorage<K, V>::kv_->Insert(key, value_ptr);
  }
 public:
  friend class HbmDramStorage<K, V>;
  friend class SetAssociativeHbmDramStorage<K, V>;
  friend class HbmDramSsdStorage<K, V>;
 protected:
  void SetTotalDims(int64 total_dims) override {}

  void Shrink(std::vector<K>& key_list,
              std::vector<ValuePtr<V>*>& value_ptr_list,
              ShrinkArgs& shrink_args,
              int64 value_len) override {
    SingleTierStorage<K, V>::Shrink(
        key_list,
        value_ptr_list,
        shrink_args,
        value_len);
  }
};

template<typename K, typename V>
class SetAssociativeHbmStorage: public SingleTierStorage<K, V> {
 public:
  SetAssociativeHbmStorage(const StorageConfig& sc, Allocator* alloc,
      LayoutCreator<V>* lc) : SingleTierStorage<K, V>(
          sc, alloc, new LocklessHashMap<K, V>(), lc) {
  }

  ~SetAssociativeHbmStorage() override {
    gpu_alloc_->DeallocateRaw(keys_);
    gpu_alloc_->DeallocateRaw(vals_);
    gpu_alloc_->DeallocateRaw(slot_counter_);
    gpu_alloc_->DeallocateRaw(global_counter_);
    gpu_alloc_->DeallocateRaw(set_mutex_);
  }

  void Init(Allocator* alloc, int cache_capacity, int alloc_len) {
    num_slot_ = cache_capacity;
    capacity_in_set_ = num_slot_ / (SET_ASSOCIATIVITY * WARP_SIZE);
    embedding_vec_size_ = alloc_len;

    gpu_alloc_ = alloc;

    // Allocate GPU memory for cache
    keys_ = (slabset *)gpu_alloc_->AllocateRaw(
      Allocator::kAllocatorAlignment,
      sizeof(slabset) * capacity_in_set_);
    vals_ = (V *)gpu_alloc_->AllocateRaw(
      Allocator::kAllocatorAlignment,
      sizeof(V) * embedding_vec_size_ * num_slot_);
    slot_counter_ = (int *)gpu_alloc_->AllocateRaw(
      Allocator::kAllocatorAlignment,
      sizeof(int) * num_slot_);
    global_counter_ = (int *)gpu_alloc_->AllocateRaw(
      Allocator::kAllocatorAlignment,
      sizeof(int));

    // Allocate GPU memory for set mutex
    set_mutex_ = (int *)gpu_alloc_->AllocateRaw(
      Allocator::kAllocatorAlignment,
      sizeof(int) * capacity_in_set_);

    test_global_counter_ = (int *)gpu_alloc_->AllocateRaw(
      Allocator::kAllocatorAlignment,
      sizeof(int));

    const K empty_key = -1;

    // Initialize the cache, set all entry to unused <K,V>
      void* args[] = {
          (void*)&keys_,
          (void*)&slot_counter_,
          (void*)&global_counter_,
          (void*)&num_slot_,
          (void*)&empty_key,
          (void*)&set_mutex_,
          (void*)&capacity_in_set_};
    cudaLaunchKernel(
      (void *)init_cache<K>,
      ((num_slot_ - 1) / BLOCK_SIZE_) + 1,
      BLOCK_SIZE_,
      args, 0, NULL);
  }
  
  void BatchGet(const EmbeddingVarContext<GPUDevice>& ctx,
                const K* keys,
                V* output,
                int64 num_of_keys,
                int64 value_len,
                int* &d_missing_index,
                K* &d_missing_keys,
                int* &d_missing_len,
                int* &d_warp_missing_counter,
                int &miss_count) {           
    // Update the global counter as user perform a new(most recent) read operation to the cache
    // Resolve distance overflow issue as well.
    void* args[] = {
        (void*)&global_counter_,
        (void*)&d_missing_len};

    cudaLaunchKernel(
      (void *)update_kernel_overflow_ignore,
      1, 1, args, 0, ctx.gpu_device.stream());

    // Read from the cache
    // Touch and refresh the hitting slot
    const int keys_per_block = (BLOCK_SIZE_ / WARP_SIZE) * task_per_warp_tile_;
    const int grid_size = ((num_of_keys - 1) / keys_per_block) + 1;

    void* args2[] = {
      (void*)&keys,
      (void*)&num_of_keys,
      (void*)&output,
      (void*)&value_len,
      (void*)&d_missing_index,
      (void*)&d_missing_keys,
      (void*)&d_missing_len,
      (void*)&global_counter_,
      (void*)&slot_counter_,
      (void*)&capacity_in_set_,
      (void*)&keys_,
      (void*)&vals_,
      (void*)&set_mutex_,
      (void*)&task_per_warp_tile_,
      (void*)&d_warp_missing_counter};

    cudaLaunchKernel(
      (void *)get_kernel<K, V>,
      grid_size,
      BLOCK_SIZE_,
      args2, 0, ctx.gpu_device.stream());

    void* args3[] = {
      (void*)&keys,
      (void*)&num_of_keys,
      (void*)&d_missing_index,
      (void*)&d_missing_keys,
      (void*)&d_missing_len,
      (void*)&d_warp_missing_counter
    };

    cudaLaunchKernel(
      (void *)get_missing_keys_and_index<K>,
      ((num_of_keys - 1) / BLOCK_SIZE_) + 1,
      BLOCK_SIZE_,
      args3, 0, ctx.gpu_device.stream());


    int N = 10000;
    void* args4[] = {
      (void*)&test_global_counter_,
      (void*)&N,
    };

    cudaLaunchKernel(
      (void *)test_add,
      (N + 127) / 128,
      128,
      args4, 0, ctx.gpu_device.stream());
  }


  void BatchGetMissing(const EmbeddingVarContext<GPUDevice>& ctx,
              const K* missing_keys,
              int* missing_index,
              V* output,
              int64 value_len,
              int miss_count,
              V *memcpy_buffer_gpu,
              bool* initialize_status,
              V *default_value_ptr) {

    /*Copy missing embeddings to output*/
    void* args[] = {
      (void*)&output,
      (void*)&memcpy_buffer_gpu,
      (void*)&missing_index,
      (void*)&value_len,
      (void*)&miss_count};

    cudaLaunchKernel(
      (void *)CopyMissingToOutput<V>,
      (miss_count * value_len - 1) / BLOCK_SIZE_ + 1,
      BLOCK_SIZE_,
      args, 0, ctx.gpu_device.stream()); 

    // Try to insert the <k,v> paris into the cache as long as there are unused slot
    // Then replace the <k,v> pairs into the cache
    const int keys_per_block = (BLOCK_SIZE_ / WARP_SIZE) * task_per_warp_tile_;
    const int grid_size = ((miss_count - 1) / keys_per_block) + 1;

    void* args2[] = {
      (void*)&missing_keys,
      (void*)&memcpy_buffer_gpu,
      (void*)&value_len,
      (void*)&miss_count,
      (void*)&keys_,
      (void*)&vals_,
      (void*)&slot_counter_,
      (void*)&set_mutex_,
      (void*)&global_counter_,
      (void*)&capacity_in_set_,
      (void*)&task_per_warp_tile_};

    cudaLaunchKernel(
      (void *)insert_replace_kernel<K, V>,
      grid_size,
      BLOCK_SIZE_,
      args2, 0, ctx.gpu_device.stream()); 
  }

  void Restore(const K* keys,
              int64 value_len,
              int size,
              V *memcpy_buffer_gpu){
    // Try to insert the <k,v> paris into the cache as long as there are unused slot
    // Then replace the <k,v> pairs into the cache
    const int keys_per_block = (BLOCK_SIZE_ / WARP_SIZE) * task_per_warp_tile_;
    const int grid_size = ((size - 1) / keys_per_block) + 1;

    void* args[] = {
      (void*)&keys,
      (void*)&memcpy_buffer_gpu,
      (void*)&value_len,
      (void*)&size,
      (void*)&keys_,
      (void*)&vals_,
      (void*)&slot_counter_,
      (void*)&set_mutex_,
      (void*)&global_counter_,
      (void*)&capacity_in_set_,
      (void*)&task_per_warp_tile_};

    cudaLaunchKernel(
      (void *)insert_replace_kernel<K, V>,
      grid_size,
      BLOCK_SIZE_,
      args, 0, NULL); 
    PrintCache();
  }

  void PrintCache(){
    cudaDeviceSynchronize();
    slabset* h_keys = nullptr;
    V* h_vals = nullptr;
    h_keys = (slabset *)malloc(sizeof(slabset) * capacity_in_set_);
    h_vals = (V *)malloc(sizeof(V) * embedding_vec_size_ * num_slot_);

    cudaMemcpy(h_keys, keys_, sizeof(slabset) * capacity_in_set_, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vals, vals_, sizeof(V) * embedding_vec_size_ * num_slot_, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 4; i++){
      for(int j = 0; j < SET_ASSOCIATIVITY; j++){
        for(int k = 0; k < WARP_SIZE; k++){
          K key = h_keys[i].set_[j].slab_[k];
          int val_index = ((i * SET_ASSOCIATIVITY + j) * WARP_SIZE + k) * embedding_vec_size_;
          if(key != -1 && key % capacity_in_set_ != i){
            LOG(INFO) << "wrong!!!!!";
          }
          std::cout << "key:" << key << std::endl;
          std::cout << "[";
          for(int m = 0; m < 10; m++){
            std::cout << h_vals[val_index + m] << ",";
          }
          std::cout << "]" << std::endl;
        }
      }
    }

    free(h_keys);
    free(h_vals);
  }

 public:
  friend class HbmDramStorage<K, V>;
  friend class SetAssociativeHbmDramStorage<K, V>;
  friend class HbmDramSsdStorage<K, V>;

  using slabset = slab_set<K>;
 protected:
  void SetTotalDims(int64 total_dims) override {}
 private:
  static const size_t BLOCK_SIZE_ = 128;
  const int task_per_warp_tile_ = 1;

  slabset* keys_ = nullptr;
  V* vals_ = nullptr;
  int* slot_counter_ = nullptr;
  int* global_counter_ = nullptr;

  // Cache capacity
  int capacity_in_set_;
  int num_slot_;

  // Embedding vector size
  int embedding_vec_size_;

  int* set_mutex_ = nullptr;

  int* test_global_counter_ = nullptr;
  
  Allocator* gpu_alloc_;
};
#endif // GOOGLE_CUDA

template<typename K, typename V>
class PmemMemkindStorage : public SingleTierStorage<K, V> {
 public:
  PmemMemkindStorage(const StorageConfig& sc, Allocator* alloc,
      LayoutCreator<V>* lc) : SingleTierStorage<K, V>(
          sc, alloc, new LocklessHashMap<K, V>(), lc) {
  }
  ~PmemMemkindStorage() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(PmemMemkindStorage);
 
 protected:
  void SetTotalDims(int64 total_dims) override {}
};

template<typename K, typename V>
class PmemLibpmemStorage : public SingleTierStorage<K, V> {
 public:
  PmemLibpmemStorage(const StorageConfig& sc, Allocator* alloc,
      LayoutCreator<V>* lc) : SingleTierStorage<K, V>(
          sc, alloc, new LocklessHashMap<K, V>(), lc) {
  }
  ~PmemLibpmemStorage() override {}

  Status Commit(K keys, const ValuePtr<V>* value_ptr) {
    return SingleTierStorage<K, V>::kv_->Commit(keys, value_ptr);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(PmemLibpmemStorage);
 
 protected:
  friend class DramPmemStorage<K, V>;
  void SetTotalDims(int64 total_dims) override {}

  void Shrink(std::vector<K>& key_list,
              std::vector<ValuePtr<V>*>& value_ptr_list,
              ShrinkArgs& shrink_args,
              int64 value_len) override {
    SingleTierStorage<K, V>::Shrink(
        key_list,
        value_ptr_list,
        shrink_args,
        value_len);
  }
};

template<typename K, typename V>
class LevelDBStore : public SingleTierStorage<K, V> {
 public:
  LevelDBStore(const StorageConfig& sc, Allocator* alloc,
      LayoutCreator<V>* lc) : SingleTierStorage<K, V>(
          sc, alloc, new LevelDBKV<K, V>(sc.path), lc) {
  }
  ~LevelDBStore() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(LevelDBStore);

  Status Commit(K keys, const ValuePtr<V>* value_ptr) {
    return SingleTierStorage<K, V>::kv_->Commit(keys, value_ptr);
  }

  embedding::ValueIterator<V>* GetValueIterator(
      const std::vector<K>& key_list,
      int64 emb_index, int64 value_len) {
    LevelDBKV<K, V>* leveldb_kv =
        reinterpret_cast<LevelDBKV<K, V>*>(SingleTierStorage<K, V>::kv_);
    return new DBValueIterator<K, V>(
        key_list, emb_index, value_len, leveldb_kv);
  }
 public:
  friend class DramLevelDBStore<K, V>;

 protected:
  void SetTotalDims(int64 total_dims) override {
    SingleTierStorage<K, V>::kv_->SetTotalDims(total_dims);
  }
};

template<typename K, typename V>
class SsdHashStorage : public SingleTierStorage<K, V> {
 public:
  SsdHashStorage(const StorageConfig& sc, Allocator* alloc,
      LayoutCreator<V>* lc) : SingleTierStorage<K, V>(
          sc, alloc, new SSDHashKV<K, V>(sc.path, alloc), lc) {
  }
  ~SsdHashStorage() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(SsdHashStorage);

  Status Commit(K keys, const ValuePtr<V>* value_ptr) {
    return SingleTierStorage<K, V>::kv_->Commit(keys, value_ptr);
  }

  Status Save(
      const string& tensor_name,
      const string& prefix,
      BundleWriter* writer,
      const EmbeddingConfig& emb_config,
      ShrinkArgs& shrink_args,
      int64 value_len,
      V* default_value) override {
    if (emb_config.is_primary()) {
      SSDHashKV<K, V>* ssd_kv =
          reinterpret_cast<SSDHashKV<K, V>*>(SingleTierStorage<K, V>::kv_);
      SsdRecordDescriptor<K> ssd_rec_desc;
      {
        mutex_lock l(Storage<K, V>::mu_);
        ssd_kv->SetSsdRecordDescriptor(&ssd_rec_desc);
      }
      ssd_rec_desc.GenerateCheckpoint(prefix, tensor_name);
    }
    return Status::OK();
  }

  void Import(K* key_list, int64* key_file_id_list,
      int64* key_offset_list, int64 num_of_keys,
      std::map<int64, int64>& file_id_map) {
    SSDHashKV<K, V>* ssd_kv =
        reinterpret_cast<SSDHashKV<K, V>*>(SingleTierStorage<K, V>::kv_);

    ssd_kv->Import(key_list, key_file_id_list,
                   key_offset_list, num_of_keys,
                   file_id_map);
  }

  void CopyEmbFilesFromCkpt(
      int64* file_list, int64* invalid_record_count_list,
      int64* record_count_list, int64 num_of_files,
      const std::string& ssd_emb_file_name) {
    SSDHashKV<K, V>* ssd_kv =
        reinterpret_cast<SSDHashKV<K, V>*>(SingleTierStorage<K, V>::kv_);

    ssd_kv->CopyEmbFilesFromCkpt(
        file_list, invalid_record_count_list,
        record_count_list, num_of_files,
        ssd_emb_file_name);
  }

  void SetSsdRecordDescriptor(SsdRecordDescriptor<K>* ssd_rec_desc) {
    SSDHashKV<K, V>* ssd_kv =
        reinterpret_cast<SSDHashKV<K, V>*>(SingleTierStorage<K, V>::kv_);
    ssd_kv->SetSsdRecordDescriptor(ssd_rec_desc);
  }
 public:
  friend class DramSsdHashStorage<K, V>;
#if GOOGLE_CUDA
  friend class HbmDramSsdStorage<K, V>;
#endif

 protected:
  void SetTotalDims(int64 total_dims) override {
    SingleTierStorage<K, V>::kv_->SetTotalDims(total_dims);
  }
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_H_
