#pragma once

#include <tuple>
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <numeric>

#include <chrono>

#if !defined(RNNT_DISABLE_OMP) && !defined(APPLE)
#include <omp.h>
#endif

#include "rnnt_helper.h"

template<typename ProbT>
class CpuRNNT {
public:
    // Noncopyable
    CpuRNNT(int minibatch, int alphabet_size, void* workspace, 
            int blank, int num_threads) :
        minibatch_(minibatch), alphabet_size_(alphabet_size), 
        workspace_(workspace), blank_(blank), num_threads_(num_threads) {
#if defined(RNNT_DISABLE_OMP) || defined(APPLE)
#else
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        } else {
            num_threads_ = omp_get_max_threads();
        }
#endif
    };

    CpuRNNT(const CpuRNNT&) = delete;
    CpuRNNT& operator=(const CpuRNNT&) = delete;

    rnntStatus_t cost_and_grad(const ProbT* const acts,
                              ProbT* grads,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);
    
    rnntStatus_t score_forward(const ProbT* const acts,
                              ProbT* costs,
                              const int* const flat_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);

private:
    class CpuRNNT_index {
    public:
        CpuRNNT_index(int U, int alphabet_size);
        int U;
        int alphabet_size;

        int operator()(int t, int u);
        int operator()(int t, int u, int v);
    };

    class CpuRNNT_metadata {
    public:
        CpuRNNT_metadata(int T, int U, void* workspace, size_t bytes_used, int alphabet_size, const ProbT* const acts, CpuRNNT_index& idx);
        ProbT* denom;
        ProbT* alphas;
        ProbT* betas;

    private:
        void setup_log_softmax_denom(const ProbT* const acts, int T, int U, int alphabet_size, CpuRNNT_index& idx);
    };

    int minibatch_;
    int alphabet_size_; // Number of characters plus blank
    void* workspace_;
    int blank_;
    int num_threads_;
    
    ProbT cost_and_grad_kernel(const ProbT* const acts, ProbT* grad,
                               const int* const labels, int mb,
                               int T, int U, size_t bytes_used);
    
    ProbT compute_alphas(const ProbT* const acts, const int* const labels, int T, int U, const ProbT* const denom, ProbT* alphas);
    
    ProbT compute_betas_and_grad(ProbT* grad, const ProbT* const acts,
                                 int T, int U, const ProbT* const denom, 
                                 ProbT* alphas, ProbT* betas,
                                 const int* const labels, ProbT logll);
};

template<typename ProbT>
CpuRNNT<ProbT>::CpuRNNT_metadata::CpuRNNT_metadata(int T, int U, void* workspace, size_t bytes_used, int alphabet_size, const ProbT* const acts, CpuRNNT_index& idx) {
    
    denom = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * T * U;

    alphas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * T * U;

    betas = reinterpret_cast<ProbT *>(static_cast<char *>(workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * T * U;

    setup_log_softmax_denom(acts, T, U, alphabet_size, idx);
}

template<typename ProbT>
void
CpuRNNT<ProbT>::CpuRNNT_metadata::setup_log_softmax_denom(const ProbT* const acts, int T, int U, int alphabet_size, CpuRNNT_index& idx) {

    for (int t = 0; t < T; ++t) {
        for (int u = 0; u < U; ++u) {
            ProbT max_v = rnnt_helper::neg_inf<ProbT>();
            for (int v = 0; v < alphabet_size; v++) {
                max_v = std::max(max_v, acts[idx(t, u, v)]);
            }

            ProbT den = rnnt_helper::neg_inf<ProbT>();
            for (int v = 0; v < alphabet_size; v++) {
                den = rnnt_helper::log_sum_exp<ProbT>(den, acts[idx(t, u, v)] - max_v);
            }
            denom[idx(t, u)] = -max_v - den;
        }
    }

#if defined(DEBUG_KERNEL)
    printf("cpu acts and denoms\n");
    for (int t = 0; t < T; t++) {
        for (int u = 0; u < U; u++) {
            for (int v = 0; v < alphabet_size; v++) {
                printf("%.4f ", acts[idx(t, u, v)]);
            }
            printf("=> %.4f; ", denom[idx(t, u)]);
        }
        printf("\n");
    }
    printf("\n");
#endif
}

template<typename ProbT>
CpuRNNT<ProbT>::CpuRNNT_index::CpuRNNT_index(int U, int alphabet_size) : 
                    U(U), alphabet_size(alphabet_size) {}

template<typename ProbT>
inline int CpuRNNT<ProbT>::CpuRNNT_index::operator()(int t, int u) {
    return t * U + u;
}

template<typename ProbT>
inline int CpuRNNT<ProbT>::CpuRNNT_index::operator()(int t, int u, int v) {
    return (t * U + u) * alphabet_size + v;
}

template<typename ProbT>
ProbT
CpuRNNT<ProbT>::cost_and_grad_kernel(const ProbT* const acts, ProbT* grad,
                              const int* const labels,
                              int mb, int T, int U, size_t bytes_used) {
    
    CpuRNNT_index idx(U, alphabet_size_);
    CpuRNNT_metadata rnntm(T, U, workspace_, bytes_used, alphabet_size_, acts, idx);

    // zero grads
    memset(grad, 0, sizeof(ProbT) * T * U * alphabet_size_);

    ProbT llForward = compute_alphas(acts, labels, T, U, rnntm.denom, rnntm.alphas);
    ProbT llBackward = compute_betas_and_grad(grad, acts, 
                                              T, U,
                                              rnntm.denom,
                                              rnntm.alphas, 
                                              rnntm.betas,
                                              labels,
                                              llForward);

    ProbT diff = std::abs(llForward - llBackward);
    if (diff > 1e-1) {
        printf("WARNING: Forward backward likelihood mismatch %f\n", diff);
    }

    return -llForward;
}

template<typename ProbT>
ProbT
CpuRNNT<ProbT>::compute_alphas(const ProbT* const acts, const int* const labels, int T, int U, const ProbT* const denom, ProbT* alphas) {

    CpuRNNT_index idx(U, alphabet_size_);

    alphas[0] = 0;
    for (int u = 1; u < U; ++u) {
        alphas[u] = rnnt_helper::neg_inf<ProbT>();
    }

    for (int t = 1; t < T; ++t) {
        alphas[idx(t, 0)] = alphas[idx(t-1, 0)] + acts[idx(t-1, 0, blank_)] + denom[idx(t-1, 0)];
        for (int u = 1; u < U; ++u) {
            ProbT no_emit = alphas[idx(t-1, u)] + acts[idx(t-1, u, blank_)] + denom[idx(t-1, u)];
            ProbT emit = alphas[idx(t-1, u-1)] + acts[idx(t-1, u-1, labels[u-1])] + denom[idx(t-1, u-1)];
            alphas[idx(t, u)] = rnnt_helper::log_sum_exp<ProbT>(emit, no_emit);
        }
    }

#ifdef DEBUG_KERNEL
    printf("cpu alphas:\n");
    printf("%d %d\n", T, U);
    for (int t = 0; t < T; t++) {
        for (int u = 0; u < U; u++) {
            printf("%.2f ", alphas[idx(t, u)]);
        }
        printf("\n");
    }
    printf("\n");
#endif

    ProbT loglike = alphas[idx(T-1, U-1)] + acts[idx(T-1, U-1, blank_)] + denom[idx(T-1, U-1)];

    return loglike;
}

template<typename ProbT>
ProbT
CpuRNNT<ProbT>::compute_betas_and_grad(ProbT* grad, const ProbT* const acts, int T, int U, 
                                       const ProbT* const denom, ProbT* alphas, ProbT* betas,
                                       const int* const labels, ProbT logll) {

    CpuRNNT_index idx(U, alphabet_size_);

    betas[idx(T-1, U-1)] = acts[idx(T-1, U-1, blank_)] + denom[idx(T-1, U-1)];
    for (int u = 0; u < U-1; ++u) {
        betas[idx(T-1, u)] = rnnt_helper::neg_inf<ProbT>();
    }

    for (int t = T-2; t >= 0; --t) {
        betas[idx(t, U-1)] = betas[idx(t+1, U-1)] + acts[idx(t, U-1, blank_)] + denom[idx(t, U-1)];
        for (int u = 0; u < U-1; ++u) {
            ProbT no_emit = betas[idx(t+1, u)] + acts[idx(t, u, blank_)] + denom[idx(t, u)];
            ProbT emit = betas[idx(t+1, u+1)] + acts[idx(t, u, labels[u])] + denom[idx(t, u)];
            betas[idx(t, u)] = rnnt_helper::log_sum_exp<ProbT>(emit, no_emit);
        }
    }

#ifdef DEBUG_KERNEL
    printf("cpu betas:\n");
    printf("%d %d\n", T, U);
    for (int t = 0; t < T; t++) {
        for (int u = 0; u < U; u++) {
            printf("%.2f ", betas[idx(t, u)]);
        }
        printf("\n");
    }
    printf("\n");
#endif


    ProbT loglike = betas[0];

    // Gradients w.r.t. log probabilities
    for (int t = 0; t < T - 1; ++t) {
        for (int u = 0; u < U; ++u) {
            for (int v = 0; v < alphabet_size_; ++v) {
                ProbT g = std::exp(acts[idx(t, u, v)] + denom[idx(t, u)] + alphas[idx(t, u)] + betas[idx(t, u)] - loglike);
                if (v == blank_) {
                    g -= std::exp(acts[idx(t, u, v)] + denom[idx(t, u)] + alphas[idx(t, u)] + betas[idx(t+1, u)] - loglike);
                } else if (u < U-1 && v == labels[u]) {
                    g -= std::exp(acts[idx(t, u, v)] + denom[idx(t, u)] + alphas[idx(t, u)] + betas[idx(t+1, u+1)] - loglike);
                }
                grad[idx(t, u, v)] = g;
            }
        }
    }

    for (int v = 0; v < alphabet_size_; ++v) {
        if (v != blank_) {
          grad[idx(T-1, U-1, v)] = std::exp(acts[idx(T-1, U-1, v)] + denom[idx(T-1, U-1)]);
        }
    }

#if defined(DEBUG_KERNEL)
        printf("cpu grads\n");
        int V = alphabet_size_;
        for (int t = 0; t < T; ++t) {
            for (int u = 0; u < U; ++u) {
                for (int v = 0; v < V; ++v) {
                    printf("%.2f ", grad[(t*U + u) * V + v]);
                }
                printf("; ");
            }
            printf("\n");
        }
        printf("\n");
#endif

    return loglike;
}

template<typename ProbT>
rnntStatus_t
CpuRNNT<ProbT>::cost_and_grad(const ProbT* const acts,
                            ProbT* grads,
                            ProbT* costs,
                            const int* const flat_labels,
                            const int* const label_lengths,
                            const int* const input_lengths) {

    size_t bytes_used[minibatch_ + 1];
    size_t start_indices[minibatch_ + 1];
    bytes_used[0] = 0;
    start_indices[0] = 0;
    int max_U = 0;

    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb];
        const int U = label_lengths[mb] + 1;
        // alphas & betas; log-softmax denom
        bytes_used[mb + 1] = bytes_used[mb] + sizeof(ProbT) * T * U * 3;
        start_indices[mb + 1] = start_indices[mb] + T * U * alphabet_size_;
        max_U = std::max(U, max_U);
    }

#pragma omp parallel for 
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb];     // Length of utterance (time)
        const int U = label_lengths[mb] + 1; // Number of labels in transcription

        costs[mb] = cost_and_grad_kernel(acts + start_indices[mb],
                             grads + start_indices[mb],
                             flat_labels + mb * (max_U - 1),
                             mb, T, U, bytes_used[mb]);
    }

    return RNNT_STATUS_SUCCESS;
}

template<typename ProbT>
rnntStatus_t
CpuRNNT<ProbT>::score_forward(const ProbT* const acts, 
                            ProbT* costs,
                            const int* const flat_labels,
                            const int* const label_lengths,
                            const int* const input_lengths) {

    size_t bytes_used[minibatch_ + 1];
    size_t start_indices[minibatch_ + 1];
    bytes_used[0] = 0;
    start_indices[0] = 0;
    int max_U = 0;

    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb];
        const int U = label_lengths[mb] + 1;
        // alphas & betas; log-softmax denom
        bytes_used[mb + 1] = bytes_used[mb] + sizeof(ProbT) * T * U * 3;
        start_indices[mb + 1] = start_indices[mb] + T * U * alphabet_size_;
        std::max(max_U, U);
    }

#pragma omp parallel for
    for (int mb = 0; mb < minibatch_; ++mb) {
        const int T = input_lengths[mb];     // Length of utterance (time)
        const int U = label_lengths[mb] + 1; // Number of labels in transcription

        CpuRNNT_index idx(U, alphabet_size_);
        CpuRNNT_metadata rnntm(T, U, workspace_, bytes_used[mb], alphabet_size_, acts + start_indices[mb], idx);

        costs[mb] = -compute_alphas(acts + start_indices[mb], flat_labels + mb * (max_U - 1), T, U, rnntm.denom, rnntm.alphas);
    }

    return RNNT_STATUS_SUCCESS;
}
