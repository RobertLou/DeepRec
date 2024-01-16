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
#include "tensorflow/core/framework/embedding/multi_tier_storage.h"
#include "tensorflow/core/framework/embedding/single_tier_storage.h"
#include "tensorflow/core/framework/embedding/hbm_storage_iterator.h"
#include "tensorflow/core/framework/embedding/intra_thread_copy_id_allocator.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {
using se::DeviceMemoryBase;
using se::Stream;

void SyncWithEventMgr(se::Stream* stream, EventMgr* event_mgr);

namespace embedding {
template<typename K, typename V>
class SetAssociativeHbmDramStorage : public MultiTierStorage<K, V> {
 public:
  SetAssociativeHbmDramStorage(const StorageConfig& sc, 
      Allocator* gpu_alloc,
      FeatureDescriptor<V>* feat_desc, const std::string& name)
      : gpu_alloc_(gpu_alloc),
        MultiTierStorage<K, V>(sc, name) {
    hbm_ = new SetAssociativeHbmStorage<K, V>(sc, feat_desc);
    hbm_feat_desc_ = feat_desc;
    dram_feat_desc_ = new FeatureDescriptor<V>(feat_desc);
    dram_ = new DramStorage<K, V>(sc, dram_feat_desc_);
  }

  ~SetAssociativeHbmDramStorage() override {
    cudaFreeHost(d_missing_keys_buffer_);
    cudaFreeHost(memcpy_buffer_gpu_);
    cudaFreeHost(h_miss_count_);

    gpu_alloc_->DeallocateRaw(d_missing_index_buffer_);
    gpu_alloc_->DeallocateRaw(d_missing_len_buffer_);

    delete hbm_;
    delete dram_;
    delete dram_feat_desc_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(SetAssociativeHbmDramStorage);

  void Init() override {
    MultiTierStorage<K, V>::cache_capacity_ =
        Storage<K, V>::storage_config_.size[0]
        / (total_dim() * sizeof(V));
    dram_feat_desc_->InitSlotInfo(hbm_feat_desc_);
    hbm_->Init(gpu_alloc_, MultiTierStorage<K, V>::cache_capacity_, total_dim());
  }

  Status Get(K key, void** value_ptr) override {

  }

  void BatchGet(const EmbeddingVarContext<GPUDevice>& ctx,
                const K* keys,
                V* output,
                void** value_ptr_list,
                int64 num_of_keys,
                int64 value_len,
                K* &d_missing_keys,
                int* &d_missing_index,
                int* &d_missing_len,
                int &miss_count) override {
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    std::string threadIdStr = oss.str();

    std::string fileloc1 = "/root/code/DeepRec/time/HBMlookup" + threadIdStr + ".txt";
    std::string fileloc2 = "/root/code/DeepRec/time/DRAMlookup" + threadIdStr + ".txt";

    std::ofstream time_file1;
    std::ofstream time_file2;

    time_file1.open(fileloc1, std::ios::app);
    time_file2.open(fileloc2, std::ios::app);

    timespec tStart, tEnd;

    clock_gettime(CLOCK_MONOTONIC, &tStart);

    if(batch_size_ < num_of_keys){
      cudaFreeHost(d_missing_keys_buffer_);
      cudaFreeHost(h_miss_count_);
      cudaFreeHost(memcpy_buffer_gpu_);
      gpu_alloc_->DeallocateRaw(d_missing_index_buffer_);
      gpu_alloc_->DeallocateRaw(d_missing_len_buffer_);

      batch_size_ = num_of_keys;

      cudaHostAlloc((void **)&d_missing_keys_buffer_, batch_size_ * sizeof(K), cudaHostAllocDefault);
      cudaHostAlloc((void **)&memcpy_buffer_gpu_, batch_size_ * value_len * sizeof(V), cudaHostAllocWriteCombined);
      cudaHostAlloc((void **)&h_miss_count_, sizeof(int), cudaHostAllocDefault);


      d_missing_index_buffer_ =
        (int*)gpu_alloc_->AllocateRaw(
          Allocator::kAllocatorAlignment,
          batch_size_ * sizeof(int));
      d_missing_len_buffer_ =
        (int*)gpu_alloc_->AllocateRaw(
          Allocator::kAllocatorAlignment,
          sizeof(int));
    }
    d_missing_index = d_missing_index_buffer_;
    d_missing_keys = d_missing_keys_buffer_;
    d_missing_len = d_missing_len_buffer_;

    hbm_->BatchGet(
      ctx, keys, output, num_of_keys, value_len, d_missing_index, d_missing_keys, d_missing_len, h_miss_count_);
    
    //cudaStreamSynchronize(ctx.gpu_device.stream());
    miss_count = *h_miss_count_;

    clock_gettime(CLOCK_MONOTONIC, &tEnd);
		time_file1 << ((double)(tEnd.tv_sec - tStart.tv_sec)*1000000000 + tEnd.tv_nsec - tStart.tv_nsec)/1000000 << std::endl;

    clock_gettime(CLOCK_MONOTONIC, &tStart);
    if(miss_count > 0){
      int num_worker_threads = ctx.worker_threads->num_threads;
    
      auto do_work = [this, d_missing_keys, value_ptr_list]
        (int64 start, int64 limit) {
        for (int64 i = start; i < limit; i++) {
          dram_->Get(d_missing_keys[i], &value_ptr_list[i]);
        }
      };

      auto worker_threads = ctx.worker_threads;
      Shard(worker_threads->num_threads,
            worker_threads->workers, miss_count,
            1000, do_work);
    }

    clock_gettime(CLOCK_MONOTONIC, &tEnd);
		time_file2 << ((double)(tEnd.tv_sec - tStart.tv_sec)*1000000000 + tEnd.tv_nsec - tStart.tv_nsec)/1000000 << std::endl;

    time_file1.close();
    time_file2.close();
  }

  void BatchGetMissing(const EmbeddingVarContext<GPUDevice>& ctx,
              const K* missing_keys,
              int* missing_index,
              V* output,
              int64 value_len,
              int miss_count,
              V **memcpy_address,
              bool* initialize_status,
              V *default_value_ptr){    

    for(int i = 0; i < miss_count; i++){
      if(!initialize_status[i]){
        memcpy(memcpy_buffer_gpu_ + i * value_len, memcpy_address[i], value_len * sizeof(V));
      }
      else{
        //LOG(INFO) << "error!";
      }
    }
    
    hbm_->BatchGetMissing(
      ctx, missing_keys, missing_index, output, value_len, miss_count, 
      memcpy_buffer_gpu_, initialize_status, default_value_ptr);
  }

  void Insert(K key, void** value_ptr) override {

  }

  void CreateAndInsert(K key, void** value_ptr,
      bool to_dram=false) override {
  }

  void UpdateValuePtr(K key, void* new_value_ptr,
                      void* old_value_ptr) override {
    
  }

  Status GetOrCreate(K key, void** value_ptr) override {
    LOG(FATAL)<<"Stroage with HBM only supports batch APIs.";
  }

  void Import(K key, V* value,
              int64 freq, int64 version,
              int emb_index) override {
    dram_->Import(key, value, freq, version, emb_index);
  }

  int total_dim() override {
    return hbm_feat_desc_->total_dim();
  }

  void ImportToHbm(K* ids, int64 size, int64 value_len, int64 emb_index) {
    V* memcpy_buffer;
    cudaHostAlloc((void **)&memcpy_buffer, size * value_len * sizeof(V), cudaHostAllocWriteCombined);
    K* keys_gpu =
        (K*)gpu_alloc_->AllocateRaw(
            Allocator::kAllocatorAlignment,
            size * sizeof(K));
    void** cpu_value_ptrs = new void*[size];
    {
      //Mutex with other Import Ops
      mutex_lock l(memory_pool_mu_);
      for (int64 i = 0; i < size; i++) {
        dram_->Get(ids[i], &cpu_value_ptrs[i]);
      }
    }
    
    LOG(INFO) << emb_index;
    LOG(INFO) << dram_feat_desc_->GetEmbedding(cpu_value_ptrs[0], emb_index);
    LOG(INFO) << hbm_feat_desc_->GetEmbedding(cpu_value_ptrs[0], emb_index);

    for (int64 i = 0; i < size; i++) {
      memcpy(memcpy_buffer + i * value_len,
        dram_feat_desc_->GetEmbedding(cpu_value_ptrs[i], emb_index),
        value_len * sizeof(V));
    }

    cudaMemcpy(keys_gpu, ids,
        size * sizeof(K), cudaMemcpyHostToDevice);
    
    hbm_->Restore(
        keys_gpu, value_len, size, memcpy_buffer);

    gpu_alloc_->DeallocateRaw(keys_gpu);
    cudaFreeHost(memcpy_buffer);
    delete[] cpu_value_ptrs;
  }

  void PrintGPUCache(){
    mutex_lock l(*hbm_->get_mutex());
    hbm_->PrintCache();
  }

  void CopyEmbeddingsFromCPUToGPU(
      int total, const K* keys,
      const std::list<int64>& copyback_cursor,
      V** memcpy_address, size_t value_len,
      void **gpu_value_ptrs, V* memcpy_buffer_gpu,
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

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<void*>* value_ptr_list) override {
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

  void CreateEmbeddingMemoryPool(
      Allocator* alloc,
      int64 value_len,
      int64 block_size) override {

  }

  void BatchEviction() override {

  }

  Status Save(
      const string& tensor_name,
      const string& prefix,
      BundleWriter* writer,
      const EmbeddingConfig& emb_config,
      ShrinkArgs& shrink_args,
      int64 value_len,
      V* default_value) override {
    LOG(INFO) << "run here";
  }

  void Restore(const std::string& name_string,
              const std::string& file_name_string,
              int64 partition_id, int64 partition_num,
              int64 value_len, bool is_incr, bool reset_version,
              const EmbeddingConfig& emb_config,
              const Eigen::GpuDevice* device,
              BundleReader* reader, EmbeddingVar<K, V>* ev,
              FilterPolicy<K, V, EmbeddingVar<K, V>>* filter) override {
    CheckpointLoader<K, V> restorer(reinterpret_cast<Storage<K, V>*>(this),
                              ev, filter, name_string, file_name_string,
                              partition_id, partition_num,
                              is_incr, reset_version, reader);
    
    restore_cache_ = CacheFactory::Create<K>(CacheStrategy::LFU, "ads");
    restorer.RestoreCkpt(emb_config, device);
    LOG(INFO) << MultiTierStorage<K, V>::cache_capacity_;
    int64 num_of_hbm_ids =
      std::min(MultiTierStorage<K, V>::cache_capacity_,
      (int64)restore_cache_->size());

    if (num_of_hbm_ids > 0) {
      K* hbm_ids = new K[num_of_hbm_ids];
      int64* hbm_freqs = new int64[num_of_hbm_ids];
      int64* hbm_versions = nullptr;
      restore_cache_->get_cached_ids(hbm_ids, num_of_hbm_ids,
                                                      hbm_versions, hbm_freqs);
      ImportToHbm(hbm_ids, num_of_hbm_ids, value_len, emb_config.emb_index);

      delete[] hbm_ids;
      delete[] hbm_freqs;
      delete restore_cache_;

    }
  }

  Status RestoreFeatures(int64 key_num, int bucket_num, int64 partition_id,
                         int64 partition_num, int64 value_len, bool is_filter,
                         bool is_incr, const EmbeddingConfig& emb_config,
                         const Eigen::GpuDevice* device,
                         FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
                         RestoreBuffer& restore_buff) override {
    Status s = filter->Restore(key_num, bucket_num, partition_id,
                               partition_num, value_len, is_filter,
                               true/*to_dram*/, is_incr, restore_buff);

    for(int i = 0; i < key_num; i++){
     ((int64*)restore_buff.freq_buffer)[i]+=1;
    }

    restore_cache_->update((K*)restore_buff.key_buffer, key_num,
                                           (int64*)restore_buff.version_buffer,
                                           (int64*)restore_buff.freq_buffer);
    
    return s;
  }

 private:
  SetAssociativeHbmStorage<K, V>* hbm_ = nullptr;
  DramStorage<K, V>* dram_ = nullptr;
  FeatureDescriptor<V>* hbm_feat_desc_ = nullptr;
  FeatureDescriptor<V>* dram_feat_desc_ = nullptr;

  BatchCache<K>* restore_cache_ = nullptr;
  EmbeddingMemoryPool<V>* embedding_mem_pool_ = nullptr;
  Allocator* gpu_alloc_;
  mutex memory_pool_mu_; //ensure thread safety of embedding_mem_pool_
  const int copyback_flag_offset_bits_ = 60;

  //Host memory buffer for each thread
  int batch_size_ = 0;
  K* d_missing_keys_buffer_ = nullptr;
  int* h_miss_count_ = nullptr;
  int* d_missing_index_buffer_ = nullptr;
  int* d_missing_len_buffer_ = nullptr;
  V* memcpy_buffer_gpu_ = nullptr;
};
} // embedding
} // tensorflow

#endif  // GOOGLE_CUDA
#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_STORAGE_H_
