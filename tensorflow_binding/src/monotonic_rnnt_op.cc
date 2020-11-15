#ifdef RNNT_ENABLE_GPU
#define EIGEN_USE_GPU
#include <cuda.h>
#endif

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/allocator.h"
#include "rnnt.h"


REGISTER_OP("MonotonicRNNT")
    .Input("acts: float32")
    .Input("labels: int32")
    .Input("input_lengths: int32")
    .Input("label_lengths: int32")
    .Attr("blank_label: int = 0")
    .Output("costs: float32")
    .Output("grads: float32");

namespace tf = tensorflow;

namespace monotonic_rnnt {

class MonotonicRNNTOpBase : public tf::OpKernel {
  public:
    explicit MonotonicRNNTOpBase(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_label", &blank_label_));
    }

    void Compute(tf::OpKernelContext* ctx) override {
        // Grab the input tensors
        const tf::Tensor* acts;
        const tf::Tensor* labels;
        const tf::Tensor* label_lengths;
        const tf::Tensor* input_lengths;
        OP_REQUIRES_OK(ctx, ctx->input("acts", &acts));
        OP_REQUIRES_OK(ctx, ctx->input("labels", &labels));
        OP_REQUIRES_OK(ctx, ctx->input("label_lengths", &label_lengths));
        OP_REQUIRES_OK(ctx, ctx->input("input_lengths", &input_lengths));

        OP_REQUIRES(ctx, acts->shape().dims() == 2,
                    tf::errors::InvalidArgument("acts is not a 2-Tensor"));
        OP_REQUIRES(ctx, labels->shape().dims() == 2,
                    tf::errors::InvalidArgument("labels is not a 2-Tensor"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(label_lengths->shape()),
                     tf::errors::InvalidArgument("label_lengths is not a vector"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(input_lengths->shape()),
                     tf::errors::InvalidArgument("input_lengths is not a vector"));

        const auto& labels_shape = labels->shape();
        const auto& acts_shape = acts->shape();
        const auto batch_size = labels_shape.dim_size(0);
        const auto maxU = labels_shape.dim_size(1) + 1;
        const auto num_classes_raw = acts_shape.dim_size(1);

        auto acts_t = acts->tensor<float, 2>();
        auto labels_t = labels->tensor<int32_t, 2>();

        OP_REQUIRES(
                ctx, tf::FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
                tf::errors::InvalidArgument("num_classes cannot exceed max int"));
        const auto alphabet_size = static_cast<int>(num_classes_raw);

        OP_REQUIRES(
                ctx, batch_size == input_lengths->dim_size(0),
                tf::errors::InvalidArgument("len(input_lengths) != batch_size.  ",
                                            "len(input_length):  ", input_lengths->dim_size(0),
                                            " batch_size: ", batch_size));
        auto input_lengths_t = input_lengths->vec<int32_t>();

        OP_REQUIRES(
                ctx, batch_size == label_lengths->dim_size(0),
                tf::errors::InvalidArgument("len(label_lengths) != batch_size.  ",
                                            "len(label_length):  ", label_lengths->dim_size(0),
                                            " batch_size: ", batch_size));
        auto label_lengths_t = label_lengths->vec<int32_t>();

        // TODO check that labels are in the alphabet?
        // Refer to line 185, we know that
        // Tensor input_lengths is in GPU, so cannot compare with CPU variable
        //for (int b = 0; b < batch_size; b++) {
        //    OP_REQUIRES(ctx, input_lengths_t(b) <= max_time,
        //                tf::errors::InvalidArgument("input_lengths(", b, ") <= ", max_time));
        //}

        tf::Tensor* costs = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("costs", input_lengths->shape(), &costs));
        auto costs_t = costs->vec<float>();

        tf::Tensor* grads = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("grads", acts->shape(), &grads));
        set_zero(grads);
        auto grads_t = grads->tensor<float, 2>();

        auto options = create_options(ctx);
        options.blank_label = blank_label_;
        options.maxU = maxU;
        
        bool use_gpu = options.loc == RNNT_GPU;

        int* T = new int[batch_size];
        int* U = new int[batch_size];
        int* start_indices = new int[batch_size + 1];
        start_indices[0] = 0;

        if (use_gpu) {
            cudaMemcpy(T, input_lengths_t.data(), sizeof(int) * batch_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(U, label_lengths_t.data(), sizeof(int) * batch_size, cudaMemcpyDeviceToHost);
            
            for (int b = 0; b < batch_size; ++b) {
                U[b] += 1;
                start_indices[b+1] = start_indices[b] + T[b] * U[b];
            }
        } else {
            const int* const input_lengths_data = input_lengths_t.data();
            const int* const label_lengths_data = label_lengths_t.data();

            for (int b = 0; b < batch_size; ++b) {
                T[b] = input_lengths_data[b];
                U[b] = label_lengths_data[b] + 1;
                start_indices[b+1] = start_indices[b] + T[b] * U[b];
            }
        }
        options.start_indices = start_indices;

        size_t workspace_size_bytes;
        auto rnnt_status = get_workspace_size(T,
                                              U,
                                              batch_size,
                                              use_gpu,
                                              &workspace_size_bytes);

        OP_REQUIRES(ctx, rnnt_status == RNNT_STATUS_SUCCESS,
                    tf::errors::Internal("monotonic_rnnt error in get_workspace_size: ",
                                         rnntGetStatusString(rnnt_status)));

        auto workspace_shape = tf::TensorShape{static_cast<int64_t>(workspace_size_bytes)};
        tf::Tensor workspace;
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_UINT8, workspace_shape, &workspace));
        auto workspace_t = workspace.flat<uint8_t>();
        
        // compute RNNT
        rnnt_status = compute_rnnt_loss(acts_t.data(),
                                        grads_t.data(),
                                        labels_t.data(),
                                        label_lengths_t.data(),
                                        input_lengths_t.data(),
                                        alphabet_size, batch_size,
                                        costs_t.data(), workspace_t.data(), options);

        OP_REQUIRES(ctx, rnnt_status == RNNT_STATUS_SUCCESS,
                    tf::errors::Internal("monotonic_rnnt error in compute_rnnt_loss: ",
                                         rnntGetStatusString(rnnt_status)));

    }
  private:
    int blank_label_;
    virtual void set_zero(tf::Tensor* t) = 0;
    virtual rnntOptions create_options(tf::OpKernelContext* ctx) = 0;
};

class MonotonicRNNTOpCPU : public MonotonicRNNTOpBase {
  public:
    explicit MonotonicRNNTOpCPU(tf::OpKernelConstruction* ctx) : MonotonicRNNTOpBase(ctx) {
    }

  private:
    void set_zero(tf::Tensor* t) override {
        t->flat<float>().setZero();
    }

    rnntOptions create_options(tf::OpKernelContext* ctx) override {
        auto options = rnntOptions{};
        options.loc = RNNT_CPU;
        options.num_threads = ctx->device()->tensorflow_cpu_worker_threads()->num_threads;
        return options;
    }
};

REGISTER_KERNEL_BUILDER(Name("MonotonicRNNT").Device(::tensorflow::DEVICE_CPU), MonotonicRNNTOpCPU);

#ifdef RNNT_ENABLE_GPU

class MonotonicRNNTOpGPU : public MonotonicRNNTOpBase {
  public:
    explicit MonotonicRNNTOpGPU(tf::OpKernelConstruction* ctx) : MonotonicRNNTOpBase(ctx) {
    }

  private:
    void set_zero(tf::Tensor* t) override {
        // here is not need
        // cudaMemset(t->flat<float>().data(), 0, t->NumElements()*sizeof(float));
    }

    rnntOptions create_options(tf::OpKernelContext* ctx) override {
        auto cuda_stream = ctx->eigen_device<Eigen::GpuDevice>().stream();
        auto options = rnntOptions{};
        options.loc = RNNT_GPU;
        options.stream = cuda_stream;
        return options;
    }
};

REGISTER_KERNEL_BUILDER(Name("MonotonicRNNT").Device(::tensorflow::DEVICE_GPU)
                        .HostMemory("costs"),
                        MonotonicRNNTOpGPU);
#undef EIGEN_USE_GPU
#endif

}
