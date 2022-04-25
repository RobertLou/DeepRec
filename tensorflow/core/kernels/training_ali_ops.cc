/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS
#include "tensorflow/core/lib/bfloat16/bfloat16.h"

#include <algorithm>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/training_ali_op_helpers.h"
#include "tensorflow/core/kernels/training_ali_ops.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/util/work_sharder.h"

#ifdef TENSORFLOW_USE_SYCL
#include "tensorflow/core/common_runtime/sycl/sycl_util.h"
#endif  // TENSORFLOW_USE_SYCL

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
using SYCLDevice = Eigen::SyclDevice;

namespace functor {
template <typename T>
struct ApplyAdagradDecay<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad,
                  bool need_decay,
                  typename TTypes<T>::ConstScalar decay_rate,
                  typename TTypes<T>::ConstScalar decay_baseline) {
    if (need_decay) {
      accum.device(d) = (accum * decay_rate()).cwiseMax(decay_baseline());
    }
    accum.device(d) += grad.square();
    var.device(d) -= grad * lr() * accum.rsqrt();
  }
};

}

template <typename TKey, typename T, typename Tstep>
class KvSparseApplyAdagradOp : public OpKernel {
 public:
  explicit KvSparseApplyAdagradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    auto locks =
        MaybeLockEmbeddingVariableInputMutexesInOrder<TKey, T>(ctx, use_exclusive_lock_, {0, 1});

    EmbeddingVar<TKey, T>* var = NULL;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 0, &var));
    core::ScopedUnref unref_var(var);
    EmbeddingVar<TKey, T>* accum = NULL;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 1, &accum));
    core::ScopedUnref unref_accum(accum);

    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, IsLegacyScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    const Tensor& global_step = ctx->input(5);
    OP_REQUIRES(
      ctx, IsLegacyScalar(global_step.shape()),
      errors::InvalidArgument(
        "global_step is not a scalar: ", global_step.shape().DebugString()));

    int64 inner_dim = 1;
    TensorShape var_shape({var->ValueLen()});
    for (int d = 0; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d + 1),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d + 1)));
      inner_dim *= grad.dim_size(d + 1);
    }

    const TKey N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      if (inner_dim > 0) {
        timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        if(var->IsHBMDRAM()) {
          timespec part_start, part_end;
          
          clock_gettime(CLOCK_MONOTONIC, &part_start);
          auto indices_flat = indices.flat<TKey>();
          auto grad_flat = grad.flat_outer_dims<T>();
          T lr_scalar = lr.scalar<T>()();
          Tstep gs = global_step.scalar<Tstep>()();
          const TKey* key_base = &indices_flat(0);
          const T* grad_base = &grad_flat(0);
          int block_dim = 128;
          int embedding_dim = var->ValueLen(); 

          clock_gettime(CLOCK_MONOTONIC, &part_end);
          LOG(INFO) << "Other time: " << ((double)(part_end.tv_sec - part_start.tv_sec) * 1000000000 + part_end.tv_nsec - part_start.tv_nsec) / 1000000 << "ms";

          clock_gettime(CLOCK_MONOTONIC, &part_start);
          TKey *key_host;
          ValuePtr<T>* value_ptr = nullptr;
          std::vector<ValuePtr<T> *> value_ptrs;
          key_host = (TKey *)malloc(sizeof(TKey) * N); 
          cudaMemcpy(key_host, key_base, sizeof(TKey) * N, cudaMemcpyDeviceToHost);
          for(int i = 0; i < N; i++){
            bool is_filter = false;
            var->LookupOrCreateKey(key_host[i], &value_ptr, &is_filter, gs);
            value_ptrs.push_back(value_ptr);
          }//Lookup ValuePtr*
          free(key_host);
          clock_gettime(CLOCK_MONOTONIC, &part_end);
          LOG(INFO) << "Lookup time: " << ((double)(part_end.tv_sec - part_start.tv_sec) * 1000000000 + part_end.tv_nsec - part_start.tv_nsec) / 1000000 << "ms";

          clock_gettime(CLOCK_MONOTONIC, &part_start);
          bool* init_flags = new bool[N]();
          T** a = new T*[N];
          T** v = new T*[N];
 
          for(int i = 0; i < N; i++){
            a[i] = accum->LookupOrCreateEmb(value_ptrs[i], init_flags[i]);
            v[i] = var->LookupOrCreateEmb(value_ptrs[i], var->GetDefaultValue(0));
          }//Get V*
          clock_gettime(CLOCK_MONOTONIC, &part_end);
          LOG(INFO) << "Get V* time: " << ((double)(part_end.tv_sec - part_start.tv_sec) * 1000000000 + part_end.tv_nsec - part_start.tv_nsec) / 1000000 << "ms";

          clock_gettime(CLOCK_MONOTONIC, &part_start);
          accum->BatchInitEmb(N, a, accum->GetDefaultValue(0), init_flags, embedding_dim);
          clock_gettime(CLOCK_MONOTONIC, &part_end);
          LOG(INFO) << "Init time: " << ((double)(part_end.tv_sec - part_start.tv_sec) * 1000000000 + part_end.tv_nsec - part_start.tv_nsec) / 1000000 << "ms";

          clock_gettime(CLOCK_MONOTONIC, &part_start);
          T **dev_a, **dev_v;
          dev_a = (T**)var->GetBuffer2(N);
          dev_v = (T**)var->GetBuffer3(N);
          cudaMemcpy(dev_a, a, sizeof(T*) * N, cudaMemcpyHostToDevice);
          cudaMemcpy(dev_v, v, sizeof(T*) * N, cudaMemcpyHostToDevice);
   
          void* args[] = { (void*)&dev_a, (void*)&dev_v, (void*)&grad_base, (void*)&lr_scalar, (void*)&embedding_dim, (void*)&N};
          cudaLaunchKernel((void *)SparseApplyAdagradGPU<T>, (N + block_dim - 1) / block_dim * embedding_dim, block_dim, args, 0, NULL);
          cudaDeviceSynchronize();

          delete[] a;
          delete[] v;
          clock_gettime(CLOCK_MONOTONIC, &part_end);
          LOG(INFO) << "apply time: " << ((double)(part_end.tv_sec - part_start.tv_sec) * 1000000000 + part_end.tv_nsec - part_start.tv_nsec) / 1000000 << "ms";

        }
        else {
          auto indices_vec = indices.vec<TKey>();
          auto grad_flat = grad.flat_outer_dims<T>();
          T lr_scalar = lr.scalar<T>()();
          Tstep gs = global_step.scalar<Tstep>()();
          auto do_work = [this, ctx, &indices_vec, var, accum, &grad_flat,
              &gs, &lr_scalar] (int64 start_i, int64 limit_i) {
            for (int64 i = start_i; i < limit_i; i++) {
              const TKey index = indices_vec(i);
              ValuePtr<T>* value_ptr = nullptr;
              bool is_filter = false;
              OP_REQUIRES_OK(ctx, var->LookupOrCreateKey(index, &value_ptr, &is_filter,
                    gs));
              if (is_filter) {
                auto a = accum->flat(value_ptr);
                auto g = grad_flat.template chip<0>(i);
                auto v = var->flat(value_ptr);
                a += g.square();
                v -= g.constant(lr_scalar) * g * a.rsqrt();
                var->Commit(index, value_ptr);
              }
            }
          };
          const int64 cost = 1000; //very unreliable estimate for cost per step.
          auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
          Shard(worker_threads.num_threads, worker_threads.workers, N, cost, do_work);
        }//IsHBM_DRAM
        clock_gettime(CLOCK_MONOTONIC, &end);
        LOG(INFO) << "Total Op time: " << ((double)(end.tv_sec - start.tv_sec) * 1000000000 + end.tv_nsec - start.tv_nsec) / 1000000 << "ms";
      }
    }
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(Tindices, T, Tstep)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceSparseApplyAdagrad")       \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices")  \
                              .TypeConstraint<Tstep>("Tstep"),       \
                          KvSparseApplyAdagradOp<Tindices, T, Tstep>);
#define REGISTER_CPU_KERNELS(T)        \
  REGISTER_KERNELS(int32, T, int32);   \
  REGISTER_KERNELS(int64, T, int32);   \
  REGISTER_KERNELS(int32, T, int64);   \
  REGISTER_KERNELS(int64, T, int64);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(Tindices, T, Tstep)                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceSparseApplyAdagrad")       \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<T>("T")                \
                              .HostMemory("lr")                      \
                              .HostMemory("global_step")             \
                              .TypeConstraint<Tindices>("Tindices")  \
                              .TypeConstraint<Tstep>("Tstep"),       \
                          KvSparseApplyAdagradOp<Tindices, T, Tstep>);
#define REGISTER_GPU_KERNELS(T)        \
  REGISTER_KERNELS(int32, T, int32);   \
  REGISTER_KERNELS(int64, T, int32);   \
  REGISTER_KERNELS(int32, T, int64);   \
  REGISTER_KERNELS(int64, T, int64);

TF_CALL_float(REGISTER_GPU_KERNELS);

#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename Device, typename TKey, typename T, bool has_l2_shrinkage>
class KvSparseApplyFtrlOp : public OpKernel {
 public:
  explicit KvSparseApplyFtrlOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    auto locks =
        MaybeLockEmbeddingVariableInputMutexesInOrder<TKey, T>(ctx, use_exclusive_lock_, {0, 1, 2});

    EmbeddingVar<TKey, T>* var_ = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 0, &var_));
    core::ScopedUnref unref_var(var_);
    EmbeddingVar<TKey, T>* accum_ = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 1, &accum_));
    core::ScopedUnref unref_accum(accum_);
    EmbeddingVar<TKey, T>* linear_ = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 2, &linear_));
    core::ScopedUnref unref_linear(linear_);

    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    const Tensor& lr = ctx->input(5);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));

    const Tensor& l1 = ctx->input(6);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(7);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l2.shape().DebugString()));
    const int lr_power_index = has_l2_shrinkage ? 9 : 8;
    const Tensor& lr_power = ctx->input(lr_power_index);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr_power.shape()) &&
                    lr_power.scalar<T>()() <= static_cast<T>(0),
                errors::InvalidArgument("lr_power is not a "
                                        "non-positive scalar: ",
                                        lr_power.shape().DebugString()));
    int64 inner_dim = 1;
    TensorShape var_shape({var_->ValueLen()});
    for (int d = 0; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d + 1),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d + 1)));
      inner_dim *= grad.dim_size(d + 1);
    }
    const int64 N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const Tensor* l2_shrinkage;
    if (has_l2_shrinkage) {
      l2_shrinkage = &ctx->input(8);
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(l2_shrinkage->shape()) &&
              l2_shrinkage->scalar<T>()() >= static_cast<T>(0),
          errors::InvalidArgument("l2 shrinkage regularization strength "
                                  "is not a non-negative scalar: ",
                                  l2_shrinkage->shape().DebugString()));
    }

    if (N > 0) {
      if (inner_dim > 0) {
        auto indices_vec = indices.vec<TKey>();
        auto grad_flat = grad.flat_outer_dims<T>();
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T l2_shrinkage_scalar;
        if (has_l2_shrinkage) {
          l2_shrinkage_scalar = l2_shrinkage->scalar<T>()();
        }
        T lr_power_scalar = lr_power.scalar<T>()();
        auto do_work = [this, ctx, inner_dim, &var_,
                       &indices_vec, &accum_, &linear_, &grad_flat,
                       &lr_scalar, &l1_scalar, &l2_scalar, &lr_power,
                       &l2_shrinkage_scalar, &lr_power_scalar]
                       (int64 start_i, int64 limit_i) {

          for (int64 i = start_i; i < limit_i; i++) {
            const TKey index = indices_vec(i);
            ValuePtr<T>* value_ptr = nullptr;
            bool is_filter = false;
            OP_REQUIRES_OK(ctx, var_->LookupOrCreateKey(index, &value_ptr, &is_filter));
            if (is_filter) {
              auto var = var_->flat(value_ptr);
              auto accum = accum_->flat(value_ptr);
              auto linear = linear_->flat(value_ptr);
              auto grad = grad_flat.template chip<0>(i);

// Use a macro to implement the computation here due to the templating of the
// eigen tensor library.
#define COMPUTE_FTRL(grad_to_use)                                              \
  auto new_accum = accum + grad_to_use.square();                               \
  if (lr_power_scalar == static_cast<T>(-0.5)) {                               \
    linear +=                                                                  \
        grad_to_use - (new_accum.sqrt() - accum.sqrt()) / lr_scalar * var;     \
  } else {                                                                     \
    linear += grad_to_use - (new_accum.pow(-lr_power_scalar) -                 \
                             accum.pow(-lr_power_scalar)) /                    \
                                lr_scalar * var;                               \
  }                                                                            \
  Eigen::Tensor<T, 0, Eigen::RowMajor, long int> linear_sqrsum =               \
            linear.square().sum().sqrt();                                      \
  T linear_norm = linear_sqrsum(0);                                            \
  if (linear_norm > l1_scalar) {                                               \
    if (lr_power_scalar == static_cast<T>(-0.5)) {                             \
       auto eta_rec = new_accum.sqrt() / lr_scalar;                            \
       auto coef = (l1_scalar - linear_norm)  /                                \
                     ((eta_rec + static_cast<T>(2) * l2_scalar) * linear_norm);\
       var = coef * linear;                                                    \
    } else {                                                                   \
      auto eta_rec = new_accum.pow(-lr_power_scalar) / lr_scalar;              \
      auto coef = (l1_scalar - linear_norm)  /                                 \
                    ((eta_rec + static_cast<T>(2) * l2_scalar) * linear_norm); \
      var = coef * linear;                                                     \
    }                                                                          \
  } else {                                                                     \
    var = var.constant(static_cast<T>(0));                                     \
  }                                                                            \
  accum += grad.square();                                                      \
  var_->Commit(index, value_ptr);                                         
              if (has_l2_shrinkage) {
                auto grad_with_shrinkage =
                    grad + static_cast<T>(2) * l2_shrinkage_scalar * var;
                COMPUTE_FTRL(grad_with_shrinkage);
              } else {
                COMPUTE_FTRL(grad);
              }
            }
          }
#undef COMPUTE_FTRL
        };

        const int64 cost = 4500; //very unreliable estimate for cost per step.
        auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
        Shard(worker_threads.num_threads, worker_threads.workers, N, cost, do_work);
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(Tindices, T)                                         \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("KvResourceSparseApplyFtrl")                                       \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<Tindices>("Tindices"),                              \
      KvSparseApplyFtrlOp<CPUDevice, Tindices, T, /*has_l2_shrinkage=*/false>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(int64, T);   \
  REGISTER_KERNELS(int32, T);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(Tindices, T)                                        \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("KvResourceSparseApplyFtrlV2")                                    \
          .Device(DEVICE_CPU)                                                \
          .TypeConstraint<T>("T")                                            \
          .TypeConstraint<Tindices>("Tindices"),                             \
      KvSparseApplyFtrlOp<CPUDevice, Tindices, T, /*has_l2_shrinkage=*/true>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(int64, T);   \
  REGISTER_KERNELS(int32, T);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename Device, typename T, typename Tstep>
class ApplyAdagradDecayOp : public OpKernel {
 public:
  explicit ApplyAdagradDecayOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
      ctx, use_exclusive_lock_, sparse, {0, 1});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
      ctx, 0, use_exclusive_lock_, false, &var));

    OP_REQUIRES(
      ctx, var.IsInitialized(),
      errors::FailedPrecondition(
        "Attempting to use uninitialized variables: ", requested_input(0)));

    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
        ctx, 1, use_exclusive_lock_, false, &accum));
    OP_REQUIRES(
      ctx, accum.IsInitialized(),
      errors::FailedPrecondition(
        "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
      ctx, var.shape().IsSameSize(accum.shape()),
      errors::InvalidArgument(
        "var and accum do not have the same shape",
        var.shape().DebugString(), " ", accum.shape().DebugString()));

    Tensor accum_decay_power;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, Tstep>(
        ctx, 2, use_exclusive_lock_, false, &accum_decay_power));
    OP_REQUIRES(
      ctx, accum_decay_power.IsInitialized(),
      errors::FailedPrecondition(
        "Attempting to use uninitialized variables: ", requested_input(2)));

    const Tensor& lr = ctx->input(3);
    OP_REQUIRES(
      ctx, IsLegacyScalar(lr.shape()),
      errors::InvalidArgument(
        "lr is not a scalar: ", lr.shape().DebugString()));

    const Tensor& decay_step = ctx->input(4);
    OP_REQUIRES(
      ctx, IsLegacyScalar(decay_step.shape()),
      errors::InvalidArgument(
        "decay_step is not a scalar: ", decay_step.shape().DebugString()));

    const Tensor& decay_rate = ctx->input(5);
    OP_REQUIRES(
      ctx, IsLegacyScalar(decay_rate.shape()),
      errors::InvalidArgument(
        "decay_rate is not a scalar: ", decay_rate.shape().DebugString()));

    const Tensor& decay_baseline = ctx->input(6);
    OP_REQUIRES(
      ctx, IsLegacyScalar(decay_baseline.shape()),
      errors::InvalidArgument(
        "init accum is not a scalar: ", decay_baseline.shape().DebugString()));

    const Tensor& global_step = ctx->input(7);
    OP_REQUIRES(
      ctx, IsLegacyScalar(global_step.shape()),
      errors::InvalidArgument(
        "global_step is not a scalar: ", global_step.shape().DebugString()));

    const Tensor& grad = ctx->input(8);
    OP_REQUIRES(
      ctx, var.shape().IsSameSize(grad.shape()),
      errors::InvalidArgument(
        "var and grad do not have the same shape",
        var.shape().DebugString(), " ", grad.shape().DebugString()));

    bool need_decay = false;
    auto accum_decay_power_flat = accum_decay_power.flat<Tstep>();
    Tstep global_step_scalar = global_step.scalar<Tstep>()();
    Tstep decay_step_scalar = decay_step.scalar<Tstep>()();
    if (global_step_scalar / decay_step_scalar > accum_decay_power_flat(0)) {
      accum_decay_power_flat(0) += 1;
      need_decay = true;
    }

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyAdagradDecay<Device, T>()(
      device, var.flat<T>(), accum.flat<T>(), lr.scalar<T>(),
      grad.flat<T>(), need_decay, decay_rate.scalar<T>(),
      decay_baseline.scalar<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T, Tstep)                                        \
  REGISTER_KERNEL_BUILDER(Name("ApplyAdagradDecay")                          \
                              .Device(DEVICE_##D)                            \
                              .TypeConstraint<T>("T")                        \
                              .TypeConstraint<Tstep>("Tstep"),               \
                          ApplyAdagradDecayOp<D##Device, T, Tstep>);         \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdagradDecay")                  \
                              .HostMemory("var")                             \
                              .HostMemory("accum")                           \
                              .HostMemory("accum_decay_power")               \
                              .Device(DEVICE_##D)                            \
                              .TypeConstraint<T>("T")                        \
                              .TypeConstraint<Tstep>("Tstep"),               \
                          ApplyAdagradDecayOp<D##Device, T, Tstep>);

#define REGISTER_CPU_KERNELS(T)        \
  REGISTER_KERNELS(CPU, T, int32);     \
  REGISTER_KERNELS(CPU, T, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename T, typename Tindex, typename Tstep>
class SparseApplyAdagradDecayOp : public OpKernel {
 public:
  explicit SparseApplyAdagradDecayOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<CPUDevice, T>(
      ctx, use_exclusive_lock_, sparse, {0, 1});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, T>(
        ctx, 0, use_exclusive_lock_, true, &var));
    OP_REQUIRES(
      ctx, var.IsInitialized(),
      errors::FailedPrecondition(
        "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
      ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
      errors::InvalidArgument("var must be at least 1 dimensional"));

    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, T>(
        ctx, 1, use_exclusive_lock_, true, &accum));
    OP_REQUIRES(
      ctx, accum.IsInitialized(),
      errors::FailedPrecondition(
        "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
      ctx, var.shape().IsSameSize(accum.shape()),
      errors::InvalidArgument(
        "var and accum do not have the same shape",
        var.shape().DebugString(), " ", accum.shape().DebugString()));

    Tensor accum_decay_power;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, Tstep>(
        ctx, 2, use_exclusive_lock_, true, &accum_decay_power));
    OP_REQUIRES(
      ctx, accum_decay_power.IsInitialized(),
      errors::FailedPrecondition(
        "Attempting to use uninitialized variables: ", requested_input(2)));

    const Tensor& lr = ctx->input(3);
    OP_REQUIRES(
      ctx, IsLegacyScalar(lr.shape()),
      errors::InvalidArgument(
        "lr is not a scalar: ", lr.shape().DebugString()));

    const Tensor& decay_step = ctx->input(4);
    OP_REQUIRES(
      ctx, IsLegacyScalar(decay_step.shape()),
      errors::InvalidArgument(
        "decay_step is not a scalar: ", decay_step.shape().DebugString()));

    const Tensor& decay_rate = ctx->input(5);
    OP_REQUIRES(
      ctx, IsLegacyScalar(decay_rate.shape()),
      errors::InvalidArgument(
        "decay_rate is not a scalar: ", decay_rate.shape().DebugString()));

    const Tensor& decay_baseline = ctx->input(6);
    OP_REQUIRES(
      ctx, IsLegacyScalar(decay_baseline.shape()),
      errors::InvalidArgument(
        "init accum is not a scalar: ", decay_baseline.shape().DebugString()));

    const Tensor& global_step = ctx->input(7);
    OP_REQUIRES(
      ctx, IsLegacyScalar(global_step.shape()),
      errors::InvalidArgument(
        "global_step is not a scalar: ", global_step.shape().DebugString()));

    const Tensor& grad = ctx->input(8);
    const Tensor& indices = ctx->input(9);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(
        ctx, var.dim_size(d) == grad.dim_size(d),
        errors::InvalidArgument(
          strings::StrCat("var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
      ctx, inner_dim > 0,
      errors::InvalidArgument("Inner dimension should be greater than zero."));

    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
      ctx, grad.dim_size(0) == N,
      errors::InvalidArgument(
        "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      auto indices_vec = indices.vec<Tindex>();
      auto accum_decay_power_flat = accum_decay_power.flat<Tstep>();
      T lr_scalar = lr.scalar<T>()();
      Tstep global_step_scalar = global_step.scalar<Tstep>()();
      Tstep decay_step_scalar = decay_step.scalar<Tstep>()();
      T decay_rate_scalar = decay_rate.scalar<T>()();
      T decay_baseline_scalar = decay_baseline.scalar<T>()();

      if (inner_dim > 1) {
        const Tindex first_dim_size = var.dim_size(0);
        auto var_flat = var.flat_outer_dims<T>();
        auto accum_flat = accum.flat_outer_dims<T>();
        auto grad_flat = grad.flat_outer_dims<T>();
        auto do_work = [this, &indices_vec, &first_dim_size, ctx,
            &accum_flat, &grad_flat, &var_flat, &global_step_scalar,
            &decay_step_scalar, &accum_decay_power_flat, &decay_rate_scalar,
            &decay_baseline_scalar, &lr_scalar] (int64 start_i, int64 limit_i) {
          for (Tindex i = start_i; i < limit_i; i++) {
            const Tindex index = internal::SubtleMustCopy(indices_vec(i));
            OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                        errors::InvalidArgument(
                            strings::StrCat("Index ", index, " at offset ", i,
                                            " in indices is out of range")));
            auto a = accum_flat.template chip<0>(index);
            auto g = grad_flat.template chip<0>(i);
            auto v = var_flat.template chip<0>(index);
            if (global_step_scalar / decay_step_scalar > accum_decay_power_flat(index)) {
              a *= a.constant(decay_rate_scalar);
              a = a.cwiseMax(decay_baseline_scalar);
              accum_decay_power_flat(index) += 1;
            }
            a += g.square();
            v -= g.constant(lr_scalar) * g * a.rsqrt();
          }
        };
        const int64 cost = 1000;
        auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
        Shard(worker_threads.num_threads, worker_threads.workers, N, cost, do_work);
      } else {
        auto var_flat = var.flat<T>();
        auto accum_flat = accum.flat<T>();
        auto grad_flat = grad.flat<T>();
        const Tindex first_dim_size = accum_flat.size();
        auto do_work = [this, ctx, &indices_vec, &first_dim_size, &accum_flat, &grad_flat,
            &global_step_scalar, &decay_step_scalar, &accum_decay_power_flat,
            &decay_rate_scalar, &decay_baseline_scalar, &lr_scalar, &var_flat]
                (int64 start_i, int64 limit_i) {
          for (Tindex i = start_i; i < limit_i; i++) {
            const Tindex index = internal::SubtleMustCopy(indices_vec(i));
            OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                        errors::InvalidArgument(
                            strings::StrCat("Index ", index, " at offset ", i,
                                            " in indices is out of range")));
            T& a = accum_flat(index);
            const T& g = grad_flat(i);
            if (global_step_scalar / decay_step_scalar > accum_decay_power_flat(index)) {
              a *= decay_rate_scalar;
              if (a < decay_baseline_scalar) {
                a = decay_baseline_scalar;
              }
              accum_decay_power_flat(index) += 1;
            }
            a += g * g;
            var_flat(index) -= lr_scalar * g / Eigen::numext::sqrt(a);
          }
        };
        const int64 cost = 1000;
        auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
        Shard(worker_threads.num_threads, worker_threads.workers, N, cost, do_work);
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices, Tstep)                                      \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyAdagradDecay")                         \
                              .Device(DEVICE_CPU)                                 \
                              .TypeConstraint<T>("T")                             \
                              .TypeConstraint<Tindices>("Tindices")               \
                              .TypeConstraint<Tstep>("Tstep"),                    \
                          SparseApplyAdagradDecayOp<T, Tindices, Tstep>);         \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyAdagradDecay")                 \
                              .Device(DEVICE_CPU)                                 \
                              .HostMemory("var")                                  \
                              .HostMemory("accum")                                \
                              .HostMemory("accum_decay_power")                    \
                              .TypeConstraint<T>("T")                             \
                              .TypeConstraint<Tindices>("Tindices")               \
                              .TypeConstraint<Tstep>("Tstep"),                    \
                          SparseApplyAdagradDecayOp<T, Tindices, Tstep>);

#define REGISTER_CPU_KERNELS(T)        \
  REGISTER_KERNELS(T, int32, int32);   \
  REGISTER_KERNELS(T, int32, int64);   \
  REGISTER_KERNELS(T, int64, int32);   \
  REGISTER_KERNELS(T, int64, int64);   \

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename T, typename Tindex, typename Tstep>
class KvSparseApplyAdagradDecayOp : public OpKernel {
 public:
  explicit KvSparseApplyAdagradDecayOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockEmbeddingVariableInputMutexesInOrder<Tindex, T>(
      ctx, use_exclusive_lock_, {0, 1, 2});

    EmbeddingVar<Tindex, T>* var = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 0, &var));
    core::ScopedUnref unref_var(var);

    EmbeddingVar<Tindex, T>* accum = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 1, &accum));
    core::ScopedUnref unref_accum(accum);

    EmbeddingVar<Tindex, T>* accum_decay_power_var = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 2, &accum_decay_power_var));
    core::ScopedUnref unref_accum_decay_power_var(accum_decay_power_var);

    const Tensor& lr = ctx->input(3);
    OP_REQUIRES(
      ctx, IsLegacyScalar(lr.shape()),
      errors::InvalidArgument(
        "lr is not a scalar: ", lr.shape().DebugString()));

    const Tensor& decay_step = ctx->input(4);
    OP_REQUIRES(
      ctx, IsLegacyScalar(decay_step.shape()),
      errors::InvalidArgument(
        "decay_step is not a scalar: ", decay_step.shape().DebugString()));

    const Tensor& decay_rate = ctx->input(5);
    OP_REQUIRES(
      ctx, IsLegacyScalar(decay_rate.shape()),
      errors::InvalidArgument(
        "decay_rate is not a scalar: ", decay_rate.shape().DebugString()));

    const Tensor& decay_baseline = ctx->input(6);
    OP_REQUIRES(
      ctx, IsLegacyScalar(decay_baseline.shape()),
      errors::InvalidArgument(
        "init accum is not a scalar: ", decay_baseline.shape().DebugString()));

    const Tensor& global_step = ctx->input(7);
    OP_REQUIRES(
      ctx, IsLegacyScalar(global_step.shape()),
      errors::InvalidArgument(
        "global_step is not a scalar: ", global_step.shape().DebugString()));

    const Tensor& grad = ctx->input(8);
    const Tensor& indices = ctx->input(9);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    TensorShape var_shape({var->ValueLen()});
    for (int d = 0; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d + 1),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d + 1)));
      inner_dim *= grad.dim_size(d + 1);
    }
    OP_REQUIRES(
      ctx, inner_dim > 0,
      errors::InvalidArgument("Inner dimension should be greater than zero."));

    const int64 N = indices.dim_size(0);
    OP_REQUIRES(
      ctx, grad.dim_size(0) == N,
      errors::InvalidArgument(
        "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      auto indices_vec = indices.vec<Tindex>();
      T lr_scalar = lr.scalar<T>()();
      Tstep gs = global_step.scalar<Tstep>()();
      Tstep decay_step_scalar = decay_step.scalar<Tstep>()();
      T decay_rate_scalar = decay_rate.scalar<T>()();
      T decay_baseline_scalar = decay_baseline.scalar<T>()();

      if (inner_dim > 0) {
        auto grad_flat = grad.flat_outer_dims<T>();
        auto do_work = [this, ctx, &indices_vec, &var, &accum, &gs,
            &grad_flat, accum_decay_power_var, &decay_step_scalar,
            &decay_rate_scalar, &decay_baseline_scalar, &lr_scalar]
                (int64 start_i, int64 limit_i) {
          for (int64 i = start_i; i < limit_i; i++) {
            const Tindex index = indices_vec(i);
            ValuePtr<T>* value_ptr = nullptr;
            bool is_filter = false;
            OP_REQUIRES_OK(ctx, var->LookupOrCreateKey(index, &value_ptr, &is_filter, gs));
            if (is_filter) {
              auto a = accum->flat(value_ptr);

              auto g = grad_flat.template chip<0>(i);

              auto v = var->flat(value_ptr);
              auto accum_decay_power = accum_decay_power_var->flat(value_ptr);

              if (gs / decay_step_scalar > accum_decay_power(0)) {
                a *= a.constant(decay_rate_scalar);
                a = a.cwiseMax(decay_baseline_scalar);
                accum_decay_power(0) += 1;
              }
              a += g.square();
              v -= g.constant(lr_scalar) * g * a.rsqrt();
              var->Commit(index, value_ptr);
            }
          }
        };
        const int64 cost = 1000;
        auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
        Shard(worker_threads.num_threads, worker_threads.workers, N, cost, do_work);
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices, Tstep)                               \
  REGISTER_KERNEL_BUILDER(Name("KvResourceSparseApplyAdagradDecay")        \
                              .Device(DEVICE_CPU)                          \
                              .HostMemory("var")                           \
                              .HostMemory("accum")                         \
                              .HostMemory("accum_decay_power")             \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<Tindices>("Tindices")        \
                              .TypeConstraint<Tstep>("Tstep"),             \
                          KvSparseApplyAdagradDecayOp<T, Tindices, Tstep>);

#define REGISTER_CPU_KERNELS(T)        \
  REGISTER_KERNELS(T, int64, int32);   \
  REGISTER_KERNELS(T, int64, int64);   \
  REGISTER_KERNELS(T, int32, int32);   \
  REGISTER_KERNELS(T, int32, int64);   \

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class KvSparseApplyAdamOp : public OpKernel {
 public:
  explicit KvSparseApplyAdamOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockEmbeddingVariableInputMutexesInOrder<Tindex, T>(ctx, use_exclusive_lock_,
                                                      {0, 1, 2});
    EmbeddingVar<Tindex, T>* var = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 0, &var));
    core::ScopedUnref unref_var(var);

    EmbeddingVar<Tindex, T>* m = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 1, &m));
    core::ScopedUnref unref_m(m);

    EmbeddingVar<Tindex, T>* v = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 2, &v));
    core::ScopedUnref unref_v(v);

    const Tensor& beta1_power = ctx->input(3);
    const Tensor& beta2_power = ctx->input(4);
    const Tensor& lr = ctx->input(5);
    const Tensor& beta1 = ctx->input(6);
    const Tensor& beta2 = ctx->input(7);
    const Tensor& epsilon = ctx->input(8);
    const Tensor& grad = ctx->input(9);
    const Tensor& indices = ctx->input(10);
    const Tensor& global_step = ctx->input(11);

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
        errors::InvalidArgument("beta1_power is not a scalar: ",
                                beta1_power.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
        errors::InvalidArgument("beta2_power is not a scalar: ",
                                beta2_power.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(lr.shape()),
        errors::InvalidArgument("lr is not a scalar: ",
                                lr.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(beta1.shape()),
        errors::InvalidArgument("beta1 is not a scalar: ",
                                beta1.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(beta2.shape()),
        errors::InvalidArgument("beta2 is not a scalar: ",
                                beta2.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
        errors::InvalidArgument("epsilon is not a scalar: ",
                                epsilon.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(indices.shape()),
        errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    TensorShape var_shape({var->ValueLen()});
    for (int d = 0; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d + 1),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d + 1)));
      inner_dim *= grad.dim_size(d + 1);
    }
    OP_REQUIRES(
        ctx, inner_dim > 0,
        errors::InvalidArgument(
            "Inner dimension should be greater than zero."));

    OP_REQUIRES(
      ctx, IsLegacyScalar(global_step.shape()),
      errors::InvalidArgument(
        "global_step is not a scalar: ", global_step.shape().DebugString()));

    const int64 N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      T beta1_power_scalar = beta1_power.scalar<T>()();
      T beta2_power_scalar = beta2_power.scalar<T>()();
      T lr_scalar = lr.scalar<T>()();
      T beta1_scalar = beta1.scalar<T>()();
      T beta2_scalar = beta2.scalar<T>()();
      T epsilon_scalar = epsilon.scalar<T>()();
      const T alpha = lr_scalar *
          Eigen::numext::sqrt(static_cast<T>(1) - beta2_power_scalar) /
          (static_cast<T>(1) - beta1_power_scalar);

      auto DoWork = [this, ctx, inner_dim, &var, &m, &v, &grad, &indices,
           &beta1_power_scalar, &beta2_power_scalar, &lr_scalar, &beta1_scalar,
           &beta2_scalar, &epsilon_scalar, &alpha, &global_step] (int64 start_i, int64 limit_i) {
        if (inner_dim > 0) {
          auto grad_flat = grad.flat_outer_dims<T>();
          auto indices_vec = indices.vec<Tindex>();

          int64 gs = global_step.scalar<int64>()();

          for (int64 i = start_i; i < limit_i; i++) {
            const Tindex index = indices_vec(i);
            ValuePtr<T>* value_ptr = nullptr;
            bool is_filter =false;
            OP_REQUIRES_OK(ctx, var->LookupOrCreateKey(index, &value_ptr, &is_filter, gs));
            if (is_filter) {
              auto var_i = var->flat(value_ptr);
              auto m_a = m->flat(value_ptr);
              auto v_a = v->flat(value_ptr);

              auto g = grad_flat.template chip<0>(i);
              m_a += (g - m_a) * (static_cast<T>(1) - beta1_scalar);
              v_a += (g.square() - v_a) * (static_cast<T>(1) - beta2_scalar);
              var_i -= (m_a * alpha) / (v_a.sqrt() + epsilon_scalar);
              var->Commit(index, value_ptr);
            }
          }
        }
      };

      const int64 cost = 1000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost, DoWork);
    }
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                 \
  REGISTER_KERNEL_BUILDER(Name("KvResourceSparseApplyAdam")             \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<T>("T")                 \
                              .TypeConstraint<Tindices>("Tindices"),  \
                          KvSparseApplyAdamOp<CPUDevice, T, Tindices>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

namespace functor {
template <typename T>
struct ApplyAdamAsync<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::Flat beta1_power,
                  typename TTypes<T>::Flat beta2_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad, bool use_nesterov) {
    auto alpha = lr() * Eigen::numext::sqrt(T(1) - beta2_power(0)) /
                 (T(1) - beta1_power(0));

    // beta1 == μ
    // beta2 == ν
    // v     == n
    // var   == θ
    m.device(d) = m * beta1() + grad * (T(1) - beta1());
    v.device(d) = v * beta2() + grad.square() * (T(1) - beta2());
    if (use_nesterov) {
      var.device(d) -= ((grad * (T(1) - beta1()) + beta1() * m) * alpha) /
                       (v.sqrt() + epsilon());
    } else {
      var.device(d) -= (m * alpha) / (v.sqrt() + epsilon());
    }

    // update beta1_power && beta2_power
    beta1_power.device(d) = beta1_power * beta1();
    beta2_power.device(d) = beta2_power * beta2();
  }
};
} // namespace functor

// Note, this op works on cpu only.
template <typename Device, typename T>
class ApplyAdamAsyncOp : public OpKernel {
 public:
  explicit ApplyAdamAsyncOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
      ctx, use_exclusive_lock_, sparse, {0, 1, 2, 3, 4});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, false, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, false, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, false, &v));
    Tensor beta1_power;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 3, use_exclusive_lock_, false, &beta1_power));
    Tensor beta2_power;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 4, use_exclusive_lock_, false, &beta2_power));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    OP_REQUIRES(
        ctx, beta1_power.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(3)));
    OP_REQUIRES(
        ctx, beta2_power.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(4)));

    const Tensor& lr = ctx->input(5);
    const Tensor& beta1 = ctx->input(6);
    const Tensor& beta2 = ctx->input(7);
    const Tensor& epsilon = ctx->input(8);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    const Tensor& grad = ctx->input(9);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyAdamAsync<Device, T>()(
        device, var.flat<T>(), m.flat<T>(), v.flat<T>(),
        beta1_power.flat<T>(), beta2_power.flat<T>(), lr.scalar<T>(),
        beta1.scalar<T>(), beta2.scalar<T>(), epsilon.scalar<T>(),
        grad.flat<T>(), use_nesterov_);

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

#define REGISTER_KERNELS(D, T)                                          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ApplyAdamAsync").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ApplyAdamAsyncOp<CPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdamAsync")                \
                              .HostMemory("var")                        \
                              .HostMemory("m")                          \
                              .HostMemory("v")                          \
                              .HostMemory("beta1_power")                \
                              .HostMemory("beta2_power")                \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<T>("T"),                  \
                          ApplyAdamAsyncOp<CPUDevice, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename Device, typename T, typename Tindex>
class SparseApplyAdamAsyncOp : public OpKernel {
 public:
  explicit SparseApplyAdamAsyncOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("apply_sparse_rmsprop", &apply_sparse_rmsprop_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
      ctx, use_exclusive_lock_, sparse, {0, 1, 2, 3, 4});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, true, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, true, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, true, &v));
    Tensor beta1_power;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 3, use_exclusive_lock_, true, &beta1_power));
    Tensor beta2_power;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 4, use_exclusive_lock_, true, &beta2_power));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    OP_REQUIRES(
        ctx, beta1_power.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(3)));
    OP_REQUIRES(
        ctx, beta2_power.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(4)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(m.shape()),
        errors::InvalidArgument("var and m do not have the same shape",
                                var.shape().DebugString(), " ",
                                m.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(v.shape()),
        errors::InvalidArgument("var and v do not have the same shape",
                                var.shape().DebugString(), " ",
                                v.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
        errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(5);
    const Tensor& beta1 = ctx->input(6);
    const Tensor& beta2 = ctx->input(7);
    const Tensor& epsilon = ctx->input(8);
    const Tensor& grad = ctx->input(9);
    const Tensor& indices = ctx->input(10);

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(lr.shape()),
        errors::InvalidArgument("lr is not a scalar: ",
                                lr.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(beta1.shape()),
        errors::InvalidArgument("beta1 is not a scalar: ",
                                beta1.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(beta2.shape()),
        errors::InvalidArgument("beta2 is not a scalar: ",
                                beta2.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
        errors::InvalidArgument("epsilon is not a scalar: ",
                                epsilon.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(indices.shape()),
        errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(
          ctx, var.dim_size(d) == grad.dim_size(d),
          errors::InvalidArgument(strings::StrCat(
                "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    OP_REQUIRES(
        ctx, inner_dim > 0,
        errors::InvalidArgument(
            "Inner dimension should be greater than zero."));

    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      if (apply_sparse_rmsprop_) {
        const Tindex first_dim_size = var.dim_size(0);
        // Validate all the indices are in range
        auto indices_vec = indices.vec<Tindex>();
        for (Tindex i = 0; i < N; i++) {
          const Tindex index = indices_vec(i);
          OP_REQUIRES(ctx, index >= 0 && index < first_dim_size,
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                          " in indices is out of range")));
        }

        auto var_flat = var.flat_outer_dims<T>();
        auto m_flat = m.flat_outer_dims<T>();
        auto v_flat = v.flat_outer_dims<T>();
        auto grad_flat = grad.flat_outer_dims<T>();
        const T lr_scalar = lr.scalar<T>()();
        const T beta1_scalar = beta1.scalar<T>()();
        const T beta2_scalar = beta2.scalar<T>()();
        const T epsilon_scalar = epsilon.scalar<T>()();
        auto do_work = [this, ctx, &indices_vec, &v_flat, &m_flat,
            &grad_flat, &beta2_scalar, &beta1_scalar, &epsilon_scalar,
            &lr_scalar, &var_flat] (int64 start_i, int64 limit_i) {
          for (Tindex i = start_i; i < limit_i; i++) {
            const Tindex index = indices_vec(i);

            auto v_ = v_flat.template chip<0>(index);
            auto m_ = m_flat.template chip<0>(index);
            auto grad_ = grad_flat.template chip<0>(i);

            v_ = v_ * v_.constant(beta2_scalar) +
                  grad_.square() * grad_.constant(T(1) - beta2_scalar);
            m_ = m_ * m_.constant(beta1_scalar) +
                   (v_ + v_.constant(epsilon_scalar)).rsqrt() *
                       v_.constant(lr_scalar) * grad_;

            auto v = var_flat.template chip<0>(index);
            v -= m_;
          }
        };
        const int64 cost = 1000;
        auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
        Shard(worker_threads.num_threads, worker_threads.workers, N, cost, do_work);
      } else {
        auto beta1_power_flat = beta1_power.flat<T>();
        auto beta2_power_flat = beta2_power.flat<T>();
        T lr_scalar = lr.scalar<T>()();
        T beta1_scalar = beta1.scalar<T>()();
        T beta2_scalar = beta2.scalar<T>()();
        T epsilon_scalar = epsilon.scalar<T>()();
        const T alpha = lr_scalar *
            Eigen::numext::sqrt(static_cast<T>(1) - beta2_power_flat(0)) /
            (static_cast<T>(1) - beta1_power_flat(0));

        auto do_work = [this, ctx, inner_dim, &var, &m, &v, &grad, &indices,
             &beta1_power_flat, &beta2_power_flat, &lr_scalar, &beta1_scalar,
             &beta2_scalar, &epsilon_scalar, &alpha] (int64 start_i, int64 limit_i) {
          if (inner_dim > 1) {
            auto var_flat = var.flat_outer_dims<T>();
            auto m_flat = m.flat_outer_dims<T>();
            auto v_flat = v.flat_outer_dims<T>();
            auto grad_flat = grad.flat_outer_dims<T>();
            auto indices_vec = indices.vec<Tindex>();
            const Tindex first_dim_size = var.dim_size(0);

            for (Tindex i = static_cast<Tindex>(start_i); i < static_cast<Tindex>(limit_i); i++) {
              const Tindex index = internal::SubtleMustCopy(indices_vec(i));
              OP_REQUIRES(
                ctx, FastBoundsCheck(index, first_dim_size),
                errors::InvalidArgument(strings::StrCat("Index ", index,
                        " at offset ", i, " in indices is out of range")));
              auto m_a = m_flat.template chip<0>(index);
              auto v_a = v_flat.template chip<0>(index);
              auto g = grad_flat.template chip<0>(i);
              auto var_i = var_flat.template chip<0>(index);

              m_a = m_a * beta1_scalar + g * (static_cast<T>(1) - beta1_scalar);
              v_a = v_a * beta2_scalar + g.square() * (static_cast<T>(1) - beta2_scalar);
              var_i -= (m_a * alpha) / (v_a.sqrt() + epsilon_scalar);
            }
          } else {
            auto var_flat = var.flat<T>();
            auto m_flat = m.flat<T>();
            auto v_flat = v.flat<T>();
            auto grad_flat = grad.flat<T>();
            auto indices_vec = indices.vec<Tindex>();
            const Tindex first_dim_size = m_flat.size();

            for (Tindex i = static_cast<Tindex>(start_i); i < static_cast<Tindex>(limit_i); i++) {
              const Tindex index = internal::SubtleMustCopy(indices_vec(i));
              OP_REQUIRES(
                ctx, FastBoundsCheck(index, first_dim_size),
                errors::InvalidArgument(strings::StrCat("Index ", index,
                        " at offset ", i, " in indices is out of range")));
              const T& g = grad_flat(i);
              T& m_a = m_flat(index);
              T& v_a = v_flat(index);
              m_a = m_a * beta1_scalar + g * (static_cast<T>(1) - beta1_scalar);
              v_a = v_a * beta2_scalar + g * g * (static_cast<T>(1) - beta2_scalar);
              var_flat(index) -= (m_a * alpha) / (Eigen::numext::sqrt(v_a) + epsilon_scalar);
            }
          }
          beta1_power_flat(0) *= beta1_scalar;
          beta2_power_flat(0) *= beta2_scalar;
        };

        const int64 cost = 1000;
        auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
        Shard(worker_threads.num_threads, worker_threads.workers, N, cost, do_work);
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool apply_sparse_rmsprop_;
};

#define REGISTER_KERNELS(T, Tindices)                                      \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyAdamAsync")                     \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<Tindices>("Tindices"),       \
                          SparseApplyAdamAsyncOp<CPUDevice, T, Tindices>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyAdamAsync")             \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<Tindices>("Tindices"),       \
                          SparseApplyAdamAsyncOp<CPUDevice, T, Tindices>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename Device, typename T, typename Tindex, typename Tstep>
class KvSparseApplyAdamAsyncOp : public OpKernel {
 public:
  explicit KvSparseApplyAdamAsyncOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("apply_sparse_rmsprop", &apply_sparse_rmsprop_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockEmbeddingVariableInputMutexesInOrder<Tindex, T>(
      ctx, use_exclusive_lock_, {0, 1, 2, 3, 4});
    EmbeddingVar<Tindex, T>* var = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 0, &var));
    core::ScopedUnref unref_var(var);

    EmbeddingVar<Tindex, T>* m = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 1, &m));
    core::ScopedUnref unref_m(m);

    EmbeddingVar<Tindex, T>* v = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 2, &v));
    core::ScopedUnref unref_v(v);

    Tensor beta1_power;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 3, use_exclusive_lock_, true, &beta1_power));

    Tensor beta2_power;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 4, use_exclusive_lock_, true, &beta2_power));
    OP_REQUIRES(
        ctx, beta1_power.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(3)));
    OP_REQUIRES(
        ctx, beta2_power.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(4)));
    

    const Tensor& lr = ctx->input(5);
    const Tensor& beta1 = ctx->input(6);
    const Tensor& beta2 = ctx->input(7);
    const Tensor& epsilon = ctx->input(8);
    const Tensor& grad = ctx->input(9);
    const Tensor& indices = ctx->input(10);
    const Tensor& global_step = ctx->input(11);

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(lr.shape()),
        errors::InvalidArgument("lr is not a scalar: ",
                                lr.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(beta1.shape()),
        errors::InvalidArgument("beta1 is not a scalar: ",
                                beta1.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(beta2.shape()),
        errors::InvalidArgument("beta2 is not a scalar: ",
                                beta2.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
        errors::InvalidArgument("epsilon is not a scalar: ",
                                epsilon.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(indices.shape()),
        errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    TensorShape var_shape({var->ValueLen()});
    for (int d = 0; d < var_shape.dims(); d++) {
      OP_REQUIRES(
          ctx, var_shape.dim_size(d) == grad.dim_size(d + 1),
          errors::InvalidArgument(strings::StrCat(
                "var and grad must match in dimension ", d + 1)));
      inner_dim *= grad.dim_size(d + 1);
    }
    OP_REQUIRES(
        ctx, inner_dim > 0,
        errors::InvalidArgument(
            "Inner dimension should be greater than zero."));

    OP_REQUIRES(
        ctx, IsLegacyScalar(global_step.shape()),
        errors::InvalidArgument(
            "global_step is not a scalar: ", global_step.shape().DebugString()));

    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      if (apply_sparse_rmsprop_) {
        auto indices_vec = indices.vec<Tindex>();

        auto grad_flat = grad.flat_outer_dims<T>();
        const T lr_scalar = lr.scalar<T>()();
        const T beta1_scalar = beta1.scalar<T>()();
        const T beta2_scalar = beta2.scalar<T>()();
        const T epsilon_scalar = epsilon.scalar<T>()();

        auto do_work = [this, ctx, &indices_vec, &var, v, m, &grad_flat,
            &beta2_scalar, &beta1_scalar, &epsilon_scalar, &lr_scalar, &global_step]
                (int64 start_i, int64 limit_i) {
          Tstep gs = global_step.scalar<Tstep>()();
          for (Tindex i = start_i; i < limit_i; i++) {
            const Tindex index = indices_vec(i);
            ValuePtr<T>* value_ptr = nullptr;
            bool is_filter = false;
            OP_REQUIRES_OK(ctx, var->LookupOrCreateKey(index, &value_ptr, &is_filter, gs));
            if (is_filter) {
              auto v_ = v->flat(value_ptr);
              auto m_ = m->flat(value_ptr);
              auto grad_ = grad_flat.template chip<0>(i);

              v_ = v_ * v_.constant(beta2_scalar) +
              grad_.square() * grad_.constant(T(1) - beta2_scalar);
              m_ = m_ * m_.constant(beta1_scalar) +
                     (v_ + v_.constant(epsilon_scalar)).rsqrt() *
                         v_.constant(lr_scalar) * grad_;

              auto v = var->flat(value_ptr);
              v -= m_;
              var->Commit(index, value_ptr);
            }
          }
        };
        const int64 cost = 1000;
        auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
        Shard(worker_threads.num_threads, worker_threads.workers, N, cost, do_work);
      } else {
        auto beta1_power_flat = beta1_power.flat<T>();
        auto beta2_power_flat = beta2_power.flat<T>();
        T lr_scalar = lr.scalar<T>()();
        T beta1_scalar = beta1.scalar<T>()();
        T beta2_scalar = beta2.scalar<T>()();
        T epsilon_scalar = epsilon.scalar<T>()();
        const T alpha = lr_scalar *
            Eigen::numext::sqrt(static_cast<T>(1) - beta2_power_flat(0)) /
            (static_cast<T>(1) - beta1_power_flat(0));

        auto do_work = [this, ctx, inner_dim, &var, &m, &v, &grad, &indices,
             &lr_scalar, &beta1_scalar,
             &beta1_power, &beta2_power,
             &beta2_scalar, &epsilon_scalar, &alpha, &global_step] (int64 start_i, int64 limit_i) {
          ValuePtr<T>* beta1_ptr = nullptr;
          OP_REQUIRES_OK(ctx, var->LookupOrCreateKey(0, &beta1_ptr));
          auto beta1_power_flat = beta1_power.flat<T>();
          auto beta2_power_flat = beta2_power.flat<T>();

          if (inner_dim > 0) {
            auto grad_flat = grad.flat_outer_dims<T>();
            auto indices_vec = indices.vec<Tindex>();
            Tstep gs = global_step.scalar<Tstep>()();

            for (Tindex i = static_cast<Tindex>(start_i); i < static_cast<Tindex>(limit_i); i++) {
              const Tindex index = indices_vec(i);
              ValuePtr<T>* value_ptr = nullptr;
              bool is_filter = false;
              OP_REQUIRES_OK(ctx, var->LookupOrCreateKey(index, &value_ptr, &is_filter, gs));
              if (is_filter) {
                auto m_a = m->flat(value_ptr);
                auto v_a = v->flat(value_ptr);
                auto g = grad_flat.template chip<0>(i);
                auto var_i = var->flat(value_ptr);

                m_a = m_a * beta1_scalar + g * (static_cast<T>(1) - beta1_scalar);
                v_a = v_a * beta2_scalar + g.square() * (static_cast<T>(1) - beta2_scalar);
                var_i -= (m_a * alpha) / (v_a.sqrt() + epsilon_scalar);
                var->Commit(index, value_ptr);
              }
            }
          }
          beta1_power_flat(0) *= beta1_scalar;
          beta2_power_flat(0) *= beta2_scalar;
        };

        const int64 cost = 1000;
        auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
        Shard(worker_threads.num_threads, worker_threads.workers, N, cost, do_work);
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool apply_sparse_rmsprop_;
};

#define REGISTER_KERNELS(T, Tindices, Tstep)                               \
  REGISTER_KERNEL_BUILDER(Name("KvResourceSparseApplyAdamAsync")           \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<Tindices>("Tindices")        \
                              .TypeConstraint<Tstep>("Tstep"),             \
                          KvSparseApplyAdamAsyncOp<CPUDevice, T, Tindices, Tstep>);
#define REGISTER_CPU_KERNELS(T)        \
  REGISTER_KERNELS(T, int32, int32);   \
  REGISTER_KERNELS(T, int64, int32);   \
  REGISTER_KERNELS(T, int32, int64);   \
  REGISTER_KERNELS(T, int64, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename T, typename Tindex, typename Tstep>
class KvResourceSparseApplyGradientDescentOp : public OpKernel {
 public:
  explicit KvResourceSparseApplyGradientDescentOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockEmbeddingVariableInputMutexesInOrder<Tindex, T>(
      ctx, use_exclusive_lock_, {0});

    EmbeddingVar<Tindex, T>* var = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 0, &var));
    core::ScopedUnref unref_var(var);

    const Tensor& lr = ctx->input(1);
    OP_REQUIRES(
      ctx, IsLegacyScalar(lr.shape()),
      errors::InvalidArgument(
        "lr is not a scalar: ", lr.shape().DebugString()));

    const Tensor& grad = ctx->input(2);
    const Tensor& indices = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    const Tensor& global_step = ctx->input(4);
    OP_REQUIRES(
      ctx, IsLegacyScalar(global_step.shape()),
      errors::InvalidArgument(
        "global_step is not a scalar: ", global_step.shape().DebugString()));

    int64 inner_dim = 1;
    TensorShape var_shape({var->ValueLen()});
    for (int d = 0; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d + 1),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d + 1)));
      inner_dim *= grad.dim_size(d + 1);
    }
    OP_REQUIRES(
      ctx, inner_dim > 0,
      errors::InvalidArgument("Inner dimension should be greater than zero."));

    const int64 N = indices.dim_size(0);
    OP_REQUIRES(
      ctx, grad.dim_size(0) == N,
      errors::InvalidArgument(
        "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      auto indices_vec = indices.vec<Tindex>();
      T lr_scalar = lr.scalar<T>()();
      Tstep gs = global_step.scalar<Tstep>()();

      if (inner_dim > 0) {
        auto grad_flat = grad.flat_outer_dims<T>();
        auto do_work = [this, ctx, &indices_vec, var, &grad_flat, &gs,
            &lr_scalar] (int64 start_i, int64 limit_i) {
          for (int64 i = start_i; i < limit_i; i++) {
            const Tindex index = indices_vec(i);
            ValuePtr<T>* value_ptr = nullptr;
            bool is_filter = false;
            OP_REQUIRES_OK(ctx, var->LookupOrCreateKey(index, &value_ptr, &is_filter, gs));
            if (is_filter) {
              auto g = grad_flat.template chip<0>(i);
              auto v = var->flat(value_ptr);
              v -= g.constant(lr_scalar) * g;
              var->Commit(index, value_ptr);
            }
          }
        };
        const int64 cost = 1000;
        auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
        Shard(worker_threads.num_threads, worker_threads.workers, N, cost, do_work);
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices, Tstep)                               \
  REGISTER_KERNEL_BUILDER(Name("KvResourceSparseApplyGradientDescent")     \
                              .Device(DEVICE_CPU)                          \
                              .HostMemory("var")                           \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<Tindices>("Tindices")        \
                              .TypeConstraint<Tstep>("Tstep"),             \
                          KvResourceSparseApplyGradientDescentOp<T, Tindices, Tstep>);

#define REGISTER_CPU_KERNELS(T)        \
  REGISTER_KERNELS(T, int64, int32);   \
  REGISTER_KERNELS(T, int64, int64);   \
  REGISTER_KERNELS(T, int32, int32);   \
  REGISTER_KERNELS(T, int32, int64);   \

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

}  // namespace tensorflow
