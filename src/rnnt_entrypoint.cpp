#include <iostream>

#include "rnnt_entrypoint.h"
#include "cpu_rnnt.h"

#ifdef RNNT_ENABLE_GPU

#include "gpu_rnnt.h"

#endif

extern "C" {

RNNTStatus compute_rnnt_loss(RNNTWorkspaceManager<float> &workspace_manager,
                             RNNTOptions options,
                             float *costs,
                             float *gradients) {

    if (workspace_manager.get_workspace() == nullptr || costs == nullptr) {
        return RNNT_STATUS_INVALID_VALUE;
    }

    if (options.loc == RNNT_CPU) {
        CpuRNNTComputer<float> rnnt_computer(workspace_manager,
                                             options.blank_label, options.num_threads);

        if (gradients != nullptr) {
            return rnnt_computer.cost_and_grad(costs, gradients);
        } else {
            return rnnt_computer.cost(costs);
        }

    } else if (options.loc == RNNT_GPU) {
#ifdef RNNT_ENABLE_GPU
        GpuRNNTComputer<float> rnnt_computer(workspace_manager, options.blank_label, options.num_threads,
                                             options.stream);

        if (gradients != nullptr)
            return rnnt_computer.cost_and_grad(costs, gradients,);
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
