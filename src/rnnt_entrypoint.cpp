#include "status.h"
#include "rnnt_entrypoint.h"

#ifdef RNNT_ENABLE_GPU

#include "gpu_workspace_manager.h"
#include "gpu_rnnt.h"

#endif

#include "cpu_workspace_manager.h"
#include "cpu_rnnt.h"

extern "C" {

RNNTStatus compute_rnnt_loss(RNNTWorkspaceManager &workspace_manager,
                             RNNTOptions options,
                             float *costs,
                             float *gradients) {

    if (costs == nullptr) {
        return RNNT_STATUS_INVALID_VALUE;
    }

    if (options.loc == RNNT_CPU) {
        auto &cpu_workspace_manager = dynamic_cast<CpuRNNTWorkspaceManager<float> &>(workspace_manager);
        CpuRNNTComputer<float> rnnt_computer(cpu_workspace_manager, options.blank_label, options.num_threads);

        if (gradients != nullptr) {
            return rnnt_computer.cost_and_grad(costs, gradients);
        } else {
            return rnnt_computer.cost(costs);
        }

    } else if (options.loc == RNNT_GPU) {
#ifdef RNNT_ENABLE_GPU
        auto &gpu_workspace_manager = dynamic_cast<GpuRNNTWorkspaceManager<float> &>(workspace_manager);
        GpuRNNTComputer<float> rnnt_computer(gpu_workspace_manager, options.blank_label, options.stream);

        if (gradients != nullptr)
            return rnnt_computer.cost_and_grad(costs, gradients);
        else
            return rnnt_computer.cost(costs);
#else
        std::cerr << "GPU execution requested, but not compiled with GPU support" << std::endl;
        return RNNT_STATUS_EXECUTION_FAILED;
#endif
    } else {
        return RNNT_STATUS_INVALID_VALUE;
    }
}

}
