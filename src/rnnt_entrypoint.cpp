#include <cstddef>
#include <iostream>
#include <algorithm>

#include <rnnt.h>

#include "detail/cpu_rnnt.h"
#ifdef __CUDACC__
    #include "detail/gpu_rnnt.h"
#endif

extern "C" {

int get_monotonic_rnnt_version() {
    return 1;
}

const char* rnntGetStatusString(rnntStatus_t status) {
    switch (status) {
    case RNNT_STATUS_SUCCESS:
        return "no error";
    case RNNT_STATUS_MEMOPS_FAILED:
        return "cuda memcpy or memset failed";
    case RNNT_STATUS_INVALID_VALUE:
        return "invalid value";
    case RNNT_STATUS_EXECUTION_FAILED:
        return "execution failed";

    case RNNT_STATUS_UNKNOWN_ERROR:
    default:
        return "unknown error";

    }

}


rnntStatus_t compute_rnnt_loss(const float* const activations, //(B*T*U),V
                             float* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             int alphabet_size,
                             int minibatch,
                             float *costs,
                             void *workspace,
                             rnntOptions options) {

    if (activations == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr ||
        costs == nullptr ||
        workspace == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0)
        return RNNT_STATUS_INVALID_VALUE;

    if (options.loc == RNNT_CPU) {
        CpuRNNT<float> rnnt(minibatch, alphabet_size, workspace, 
                                options.blank_label, options.num_threads);

        if (gradients != NULL)
            return rnnt.cost_and_grad(activations, gradients,
                                        costs,
                                        flat_labels, label_lengths,
                                        input_lengths);
        else
            return rnnt.score_forward(activations, costs, flat_labels,
                                        label_lengths, input_lengths);
    } else if (options.loc == RNNT_GPU) {
#ifdef __CUDACC__
        GpuRNNT<float> rnnt(minibatch, options.start_indices, options.maxU, alphabet_size, workspace,
                                options.blank_label, options.num_threads, options.stream);

        if (gradients != NULL)
            return rnnt.cost_and_grad(activations, gradients,
                                        costs,
                                        flat_labels, label_lengths,
                                        input_lengths);
        else
            return rnnt.score_forward(activations, costs, flat_labels,
                                        label_lengths, input_lengths);
#else
        std::cerr << "GPU execution requested, but not compiled with GPU support" << std::endl;
        return RNNT_STATUS_EXECUTION_FAILED;
#endif
    } else {
        return RNNT_STATUS_INVALID_VALUE;
    }
}


rnntStatus_t get_workspace_size(int* T, int* U,
                               int minibatch,
                               bool gpu,
                               size_t* size_bytes,
                               size_t dtype_size)
{
    if (minibatch <= 0)
        return RNNT_STATUS_INVALID_VALUE;
    for (int mb = 0; mb < minibatch; ++mb) {
        if (T[mb] <= 0 || U[mb] <= 0)
            return RNNT_STATUS_INVALID_VALUE;
    }

    *size_bytes = 0;

    for (int mb = 0; mb < minibatch; ++mb) {
        // alphas & betas
        *size_bytes += dtype_size * T[mb] * U[mb] * 2;
        // log-softmax denominator
        *size_bytes += dtype_size * T[mb] * U[mb];
        if (gpu) {
            // forward-backward loglikelihood
            *size_bytes += dtype_size * 2;
            // start-indices
            *size_bytes += sizeof(int);
        }
    }

    return RNNT_STATUS_SUCCESS;
}

rnntStatus_t compute_rnnt_loss_fp64(const double* const activations, //(B*T*U, V)
                             double* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             int alphabet_size,
                             int minibatch,
                             double *costs,
                             void *workspace,
                             rnntOptions options) {

    if (activations == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr ||
        costs == nullptr ||
        workspace == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0)
        return RNNT_STATUS_INVALID_VALUE;

    if (options.loc == RNNT_CPU) {
        CpuRNNT<double> rnnt(minibatch, alphabet_size, workspace, 
                                options.blank_label, options.num_threads);

        if (gradients != NULL)
            return rnnt.cost_and_grad(activations, gradients,
                                        costs,
                                        flat_labels, label_lengths,
                                        input_lengths);
        else
            return rnnt.score_forward(activations, costs, flat_labels,
                                        label_lengths, input_lengths);
    } else if (options.loc == RNNT_GPU) {
#ifdef __CUDACC__
        GpuRNNT<double> rnnt(minibatch, options.start_indices, options.maxU, alphabet_size, workspace,
                                options.blank_label, options.num_threads, options.stream);

        if (gradients != NULL)
            return rnnt.cost_and_grad(activations, gradients,
                                        costs,
                                        flat_labels, label_lengths,
                                        input_lengths);
        else
            return rnnt.score_forward(activations, costs, flat_labels,
                                        label_lengths, input_lengths);
#else
        std::cerr << "GPU execution requested, but not compiled with GPU support" << std::endl;
        return RNNT_STATUS_EXECUTION_FAILED;
#endif
    } else {
        return RNNT_STATUS_INVALID_VALUE;
    }
}

}
