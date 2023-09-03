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
    gpu_alloc_->DeallocateRaw(cache_);
    gpu_alloc_->DeallocateRaw(locks_);
  }

  void Init(Allocator* alloc, int cache_capacity, int alloc_len) {
    ways = 8;
    cache_size = cache_capacity;
    cache_num = cache_size / ways;
    key_size = sizeof(K);
    header_size = sizeof(FixedLengthGPUHeader<K>);
    alloc_size = header_size + alloc_len * sizeof(V);
    gpu_alloc_ = alloc;
    cache_ = (char *)gpu_alloc_->AllocateRaw(
        Allocator::kAllocatorAlignment,
        alloc_size * cache_size);
    
    locks_ = (int *)gpu_alloc_->AllocateRaw(
        Allocator::kAllocatorAlignment,
        cache_num * sizeof(int));
      
    int block_dim = 128;
      void* args[] = {
          (void*)&locks_,
          (void*)&cache_,
          (void*)&key_size,
          (void*)&header_size,
          (void*)&alloc_size,
          (void*)&alloc_len,
          (void*)&cache_num,
          (void*)&cache_size};
    cudaLaunchKernel(
      (void *)InitEmptyCache<K, V>,
      (cache_size + block_dim - 1) / block_dim,
      block_dim,
      args, 0, NULL);
  }
  
  void BatchGet(const EmbeddingVarContext<GPUDevice>& ctx,
                const K* keys,
                V* output,
                int64 num_of_keys,
                int64 value_len,
                int &miss_count,
                K *gather_status) {
    int *dev_miss_count;
    dev_miss_count = (int *)gpu_alloc_->AllocateRaw(
        Allocator::kAllocatorAlignment,
        sizeof(int));

    int block_dim = 128;
      void* args[] = {
          (void*)&keys,
          (void*)&cache_,
          (void*)&output,
          (void*)&dev_miss_count,
          (void*)&gather_status,
          (void*)&key_size,
          (void*)&header_size,
          (void*)&alloc_size,
          (void*)&value_len,
          (void*)&ways,
          (void*)&cache_num,
          (void*)&num_of_keys};
    cudaLaunchKernel(
      (void *)GatherEmbedding<K, V>,
      (num_of_keys + block_dim - 1) / block_dim,
      block_dim,
      args, 0, ctx.gpu_device.stream());
    cudaMemcpy(&miss_count, dev_miss_count, sizeof(int), cudaMemcpyDeviceToHost);

    gpu_alloc_->DeallocateRaw(dev_miss_count);
  }


  void BatchGetMissing(const EmbeddingVarContext<GPUDevice>& ctx,
              const K* keys,
              V* output,
              int64 value_len,
              int &miss_count,
              int *missing_index_gpu,
              V *memcpy_buffer_gpu,
              bool *initialize_status_gpu,
              V* default_value_ptr) {
    int block_dim = 128;
      void* args[] = {
          (void*)&locks_,
          (void*)&keys,
          (void*)&cache_,
          (void*)&output,
          (void*)&default_value_ptr,
          (void*)&missing_index_gpu,
          (void*)&memcpy_buffer_gpu,
          (void*)&initialize_status_gpu,
          (void*)&key_size,
          (void*)&header_size,
          (void*)&alloc_size,
          (void*)&value_len,
          (void*)&ways,
          (void*)&cache_num,
          (void*)&miss_count};
    /*      
    TF_CHECK_OK(GpuLaunchKernel(
        GatherMissingEmbedding<K, V>,
        (miss_count + block_dim - 1) / block_dim,
        block_dim, 0, ctx.gpu_device.stream(),
        locks_, keys, cache_, output, 
        missing_index_gpu, memcpy_buffer_gpu,
        key_size, header_size, alloc_size,
        value_len, ways, cache_num, miss_count));  */
      
    cudaLaunchKernel(
        (void *)GatherMissingEmbedding<K, V>,
        (miss_count + block_dim - 1) / block_dim,
        block_dim,
        args, 0, ctx.gpu_device.stream()); 
  
  }

  void Restore(const K* keys,
              int64 value_len,
              int size,
              V *memcpy_buffer_gpu){
    int block_dim = 128;
      void* args[] = {
          (void*)&locks_,
          (void*)&keys,
          (void*)&cache_,
          (void*)&memcpy_buffer_gpu,
          (void*)&key_size,
          (void*)&header_size,
          (void*)&alloc_size,
          (void*)&value_len,
          (void*)&ways,
          (void*)&cache_num,
          (void*)&size};

    cudaLaunchKernel(
        (void *)RestoreEmbedding<K, V>,
        (size + block_dim - 1) / block_dim,
        block_dim,
        args, 0, NULL); 
  }

  void PrintCache(){
    char *cpu_cache; 
    cpu_cache = (char *)malloc(alloc_size * cache_size);
    cudaMemcpy(cpu_cache, cache_, alloc_size * cache_size, cudaMemcpyDeviceToHost);
    
    char *base_ptr;
    int *freq_ptr;
    K *key_ptr;
    V *value_ptr;
    for(int i = 0; i < cache_size; i++){
      base_ptr = cpu_cache + alloc_size * i;
      key_ptr = reinterpret_cast<K *>(base_ptr);
      freq_ptr = reinterpret_cast<int *>(base_ptr + key_size);
      value_ptr = reinterpret_cast<V *>(base_ptr + header_size);
      std::cout << "key:" << *key_ptr << std::endl;
      std::cout << "[";
      for(int j = 0; j < 10; j++){
        std::cout << value_ptr[j] << ",";
      }
      std::cout << "]" << std::endl;
    }
    free(cpu_cache);
  }

 public:
  friend class HbmDramStorage<K, V>;
  friend class SetAssociativeHbmDramStorage<K, V>;
  friend class HbmDramSsdStorage<K, V>;
 protected:
  void SetTotalDims(int64 total_dims) override {}
 private:
  int key_size, header_size, alloc_size;
  int cache_size, cache_num, ways;
  int *locks_;
  char *cache_;
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
