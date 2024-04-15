#ifdef RNNT_ENABLE_GPU

#define EIGEN_USE_GPU
#include <cuda.h>
#include "gpu_rnnt.h"
#include "gpu_workspace_manager.h"

#else
#include "cpu_rnnt.h"
#include "cpu_workspace_manager.h"
#endif

#include "options.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tf = tensorflow;

REGISTER_OP("MonotonicRNNT")
    .Input("acts: float32")
    .Input("labels: int32")
    .Input("input_lengths: int32")
    .Input("label_lengths: int32")
    .Attr("blank_label: int = 0")
    .Output("costs: float32")
    .Output("grads: float32")
    .SetShapeFn([](tf::shape_inference::InferenceContext *c) {
        // costs shape = (B); B is the first dimension of the labels input
        c->set_output(0, c->MakeShape({c->Dim(c->input(1), 0)}));
        // grads shape = shape of acts
        c->set_output(1, c->input(0));
        return tf::Status();
    });

namespace monotonic_rnnt {

class MonotonicRNNTOpBase : public tf::OpKernel {
   public:
    explicit MonotonicRNNTOpBase(tf::OpKernelConstruction *ctx) : tf::OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_label", &blank_label_));
    }

    void Compute(tf::OpKernelContext *ctx) override {
        // Grab the input tensors
        const tf::Tensor *acts;
        const tf::Tensor *labels;
        const tf::Tensor *label_lengths;
        const tf::Tensor *input_lengths;
        OP_REQUIRES_OK(ctx, ctx->input("acts", &acts));
        OP_REQUIRES_OK(ctx, ctx->input("labels", &labels));
        OP_REQUIRES_OK(ctx, ctx->input("label_lengths", &label_lengths));
        OP_REQUIRES_OK(ctx, ctx->input("input_lengths", &input_lengths));

        OP_REQUIRES(ctx, acts->shape().dims() == 2, tf::errors::InvalidArgument("acts is not a 2-Tensor"));
        OP_REQUIRES(ctx, labels->shape().dims() == 2, tf::errors::InvalidArgument("labels is not a 2-Tensor"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(label_lengths->shape()),
                    tf::errors::InvalidArgument("label_lengths is not a vector"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(input_lengths->shape()),
                    tf::errors::InvalidArgument("input_lengths is not a vector"));

        const auto &labels_shape = labels->shape();
        const auto &acts_shape = acts->shape();
        const auto B = labels_shape.dim_size(0);
        const auto V = acts_shape.dim_size(1);

        auto acts_t = acts->tensor<float, 2>();
        auto labels_t = labels->tensor<int32_t, 2>();

        OP_REQUIRES(ctx, tf::FastBoundsCheck(V, std::numeric_limits<int>::max()),
                    tf::errors::InvalidArgument("num_classes cannot exceed max int"));

        OP_REQUIRES(ctx, B == input_lengths->dim_size(0),
                    tf::errors::InvalidArgument("len(input_lengths) != batch_size.  ", "len(input_lengths):  ",
                                                input_lengths->dim_size(0), " batch_size: ", B));
        auto input_lengths_t = input_lengths->vec<int32_t>();

        OP_REQUIRES(ctx, B == label_lengths->dim_size(0),
                    tf::errors::InvalidArgument("len(label_lengths) != batch_size.  ", "len(label_lengths):  ",
                                                label_lengths->dim_size(0), " batch_size: ", B));
        auto label_lengths_t = label_lengths->vec<int32_t>();

        tf::Tensor *costs = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("costs", input_lengths->shape(), &costs));
        auto costs_t = costs->vec<float>();

        tf::Tensor *grads = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("grads", acts->shape(), &grads));
        set_zero(grads);

        auto grads_t = grads->tensor<float, 2>();

        auto options = create_options(ctx);
        options.blank_label = blank_label_;

        // set up workspace
        size_t workspace_size_bytes;

#ifdef RNNT_ENABLE_GPU
        GpuRNNTWorkspaceManager<float> workspace_manager(acts_t.data(), labels_t.data(), static_cast<int>(B),
                                                         input_lengths_t.data(), label_lengths_t.data(),
                                                         static_cast<int>(V));
        auto rnnt_status = workspace_manager.get_workspace_size(&workspace_size_bytes, options.stream);
#else
        CpuRNNTWorkspaceManager<float> workspace_manager(acts_t.data(), labels_t.data(), static_cast<int>(B),
                                                         input_lengths_t.data(), label_lengths_t.data(),
                                                         static_cast<int>(V));
        auto rnnt_status = workspace_manager.get_workspace_size(&workspace_size_bytes);
#endif

        OP_REQUIRES(
            ctx, rnnt_status == RNNT_STATUS_SUCCESS,
            tf::errors::Internal("monotonic_rnnt error in get_workspace_size: ", rnntGetStatusString(rnnt_status)));

        auto workspace_shape = tf::TensorShape{static_cast<int64_t>(workspace_size_bytes)};
        tf::Tensor workspace;
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_UINT8, workspace_shape, &workspace));
        auto workspace_t = workspace.flat<uint8_t>();

        // compute RNNT
#ifdef RNNT_ENABLE_GPU
        workspace_manager.set_workspace(workspace_t.data(), options.stream);

        GpuRNNTComputer<float> rnnt_computer(workspace_manager, options.blank_label, options.stream);
        rnnt_status = rnnt_computer.cost_and_grad(costs_t.data(), grads_t.data());

#else
        workspace_manager.set_workspace(workspace_t.data());

        CpuRNNTComputer<float> rnnt_computer(workspace_manager, options.blank_label, options.num_threads);

        rnnt_status = rnnt_computer.cost_and_grad(costs_t.data(), grads_t.data());

#endif

        OP_REQUIRES(ctx, rnnt_status == RNNT_STATUS_SUCCESS,
                    tf::errors::Internal("monotonic_rnnt error in rnnt_computer: ", rnntGetStatusString(rnnt_status)));
    }

   private:
    int blank_label_ = 0;

    virtual void set_zero(tf::Tensor *t) = 0;

    virtual RNNTOptions create_options(tf::OpKernelContext *ctx) = 0;
};

#ifdef RNNT_ENABLE_GPU

class MonotonicRNNTOpGPU : public MonotonicRNNTOpBase {
   public:
    explicit MonotonicRNNTOpGPU(tf::OpKernelConstruction *ctx) : MonotonicRNNTOpBase(ctx) {}

   private:
    void set_zero(tf::Tensor *t) override {
        // here is no need
        // cudaMemset(t->flat<float>().data(), 0, t->NumElements()*sizeof(float));
    }

    RNNTOptions create_options(tf::OpKernelContext *ctx) override {
        auto options = RNNTOptions{};
        options.stream = ctx->eigen_device<Eigen::GpuDevice>().stream();
        return options;
    }
};

REGISTER_KERNEL_BUILDER(Name("MonotonicRNNT").Device(::tensorflow::DEVICE_GPU).HostMemory("costs"), MonotonicRNNTOpGPU);
#else

class MonotonicRNNTOpCPU : public MonotonicRNNTOpBase {
   public:
    explicit MonotonicRNNTOpCPU(tf::OpKernelConstruction *ctx) : MonotonicRNNTOpBase(ctx) {}

   private:
    void set_zero(tf::Tensor *t) override { t->flat<float>().setZero(); }

    RNNTOptions create_options(tf::OpKernelContext *ctx) override {
        auto options = RNNTOptions{};
        options.num_threads = ctx->device()->tensorflow_cpu_worker_threads()->num_threads;
        return options;
    }
};

REGISTER_KERNEL_BUILDER(Name("MonotonicRNNT").Device(::tensorflow::DEVICE_CPU), MonotonicRNNTOpCPU);
#endif

}  // namespace monotonic_rnnt
