#ifdef RNNT_ENABLE_GPU

#include "gpu_rnnt.h"
#include "gpu_workspace_manager.h"
extern THCSTATE* state;

#endif

#include <torch/extension.h>

#include "cpu_rnnt.h"
#include "cpu_workspace_manager.h"
#include "options.h"

int cpu_monotonic_rnnt(torch::Tensor acts, torch::Tensor labels, torch::Tensor input_lengths,
                       torch::Tensor label_lengths, torch::Tensor costs, torch::Tensor grads, int blank_label,
                       int num_threads) {
    TORCH_CHECK(acts.type().scalarType() == torch::ScalarType::Float);

    int B = labels.size(0);
    int V = acts.size(1);

    RNNTOptions options;
    options.loc = RNNT_CPU;
    options.blank_label = blank_label;
    options.num_threads = num_threads;

    CpuRNNTWorkspaceManager<float> workspace_manager(acts.data<float>(), labels.data<int>(), static_cast<int>(B),
                                                     input_lengths.data<int>(), label_lengths.data<int>(),
                                                     static_cast<int>(V));
    auto rnnt_status = workspace_manager.create_workspace();

    TORCH_CHECK(rnnt_status == RNNT_STATUS_SUCCESS, "cpu_rnnt error in create_workspace");

    CpuRNNTComputer<float> rnnt_computer(workspace_manager, options.blank_label, options.num_threads);

    rnnt_status = rnnt_computer.cost_and_grad(costs.data<float>(), grads.data<float>());
    TORCH_CHECK(rnnt_status == RNNT_STATUS_SUCCESS, "cpu_rnnt error in rnnt_computer");

    return rnnt_status;
}

#ifdef RNNT_ENABLE_GPU

int gpu_monotonic_rnnt(torch::Tensor acts, torch::Tensor labels, torch::Tensor input_lengths,
                       torch::Tensor label_lengths, torch::Tensor costs, torch::Tensor grads, int blank_label,
                       int num_threads) {
    TORCH_CHECK(acts.type().scalarType() == torch::ScalarType::Float);
    TORCH_CHECK(acts.type().is_cuda(), "acts must be a CUDA tensor");
    TORCH_CHECK(labels.type().is_cuda(), "acts must be a CUDA tensor");
    TORCH_CHECK(input_lengths.type().is_cuda(), "acts must be a CUDA tensor");
    TORCH_CHECK(label_lengths.type().is_cuda(), "acts must be a CUDA tensor");

    int B = labels.size(0);
    int V = acts.size(1);

    RNNTOptions options;
    options.loc = RNNT_CPU;
    options.blank_label = blank_label;
    options.stream = at::cuda::getCurrentCUDAStream();
    options.num_threads = num_threads;

    GpuRNNTWorkspaceManager<float> workspace_manager(acts.data<float>(), labels.data<int>(), static_cast<int>(B),
                                                     input_lengths.data<int>(), label_lengths.data<int>(),
                                                     static_cast<int>(V));

    size_t gpu_size_bytes;
    auto rnnt_status = workspace_manager.get_workspace_size(&gpu_size_bytes, options.stream);

    TORCH_CHECK(rnnt_status, "gpu_rnnt error in get_workspace_size");

    void* gpu_workspace = THCudaMalloc(state, gpu_size_bytes);
    workspace_manager.set_workspace(gpu_workspace, options.stream);

    GpuRNNTComputer<float> rnnt_computer(workspace_manager, options.blank_label, options.stream);
    rnnt_status = rnnt_computer.cost_and_grad(costs.data<float>(), grads.data<float>());
    TORCH_CHECK(rnnt_status, "gpu_rnnt error in rnnt_computer");

    THCudaFree(state, gpu_workspace);

    return rnnt_status;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu_monotonic_rnnt", &cpu_monotonic_rnnt, "Monotonic RNNT CPU version");
#ifdef RNNT_ENABLE_GPU
    m.def("gpu_monotonic_rnnt", &gpu_monotonic_rnnt, "Monotonic RNNT GPU version");
#endif
}
