#ifndef MONOTONIC_RNNT_CPU_RNNT_H
#define MONOTONIC_RNNT_CPU_RNNT_H

#include <tuple>
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <numeric>

#include <chrono>

#ifndef RNNT_DISABLE_OMP

#include <omp.h>

#endif

#include "rnnt_helper.h"
#include "workspace_manager.h"

template<typename ProbT>
class CpuRNNTComputer {
public:
    // Noncopyable
    CpuRNNTComputer(RNNTWorkspaceManager<ProbT> &workspace_manager, int blank, int num_threads) :
            workspace_manager_(workspace_manager), blank_(blank) {
#ifndef RNNT_DISABLE_OMP
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        } else {
            omp_get_max_threads();
        }
#endif
    };

    CpuRNNTComputer(const CpuRNNTComputer &) = delete;

    CpuRNNTComputer &operator=(const CpuRNNTComputer &) = delete;

    RNNTStatus cost_and_grad(ProbT *costs, ProbT *grads);

    RNNTStatus cost(ProbT *costs);

private:

    RNNTWorkspaceManager<ProbT> &workspace_manager_;
    int blank_;

    void setup_log_softmax_denom();

    ProbT compute_alphas_kernel(int b);

    ProbT cost_and_grad_kernel(int b, ProbT *grad);

    ProbT compute_betas_and_grad_kernel(int b, ProbT *grad);

};

template<typename ProbT>
void CpuRNNTComputer<ProbT>::setup_log_softmax_denom() {

#pragma omp parallel for default(none)
    for (int b = 0; b < workspace_manager_.B(); ++b) {
        for (int t = 0; t < workspace_manager_.T(b); ++t) {
            for (int s = 0; s <= workspace_manager_.S(b); ++s) {
                ProbT max_v = -std::numeric_limits<ProbT>::infinity();
                for (int v = 0; v < workspace_manager_.V(); ++v) {
                    max_v = std::max(max_v, workspace_manager_.act(b, t, s, v));
                }

                ProbT den = -std::numeric_limits<ProbT>::infinity();
                for (int v = 0; v < workspace_manager_.V(); v++) {
                    den = rnnt_helper::log_sum_exp<ProbT>(den, workspace_manager_.act(b, t, s, v) - max_v);
                }
                workspace_manager_.set_denom(b, t, s, -max_v - den);
            }
        }

    }

#ifdef DEBUG_KERNEL
    printf("cpu acts and denominators\n");
    for (int b = 0; b < workspace_manager_.B(); ++b) {
        printf("b = %d\n", b);
        for (int t = 0; t < workspace_manager_.T(b); ++t) {
            printf("  t = %d\n", t);
            for (int s = 0; s <= workspace_manager_.S(b); ++s) {
                printf("    s = %d\n      ", s);
                for (int v = 0; v < workspace_manager_.V(); ++v) {
                    printf("%.4f ", workspace_manager_.act(b, t, s, v));
                }
                printf(" => %.4f\n", workspace_manager_.get_denom(b, t, s));
            }
        }
    }
    printf("\n");
#endif
}

template<typename ProbT>
ProbT
CpuRNNTComputer<ProbT>::compute_alphas_kernel(int b) {
    for (int t = 0; t < workspace_manager_.T(b); ++t) {
        for (int s = workspace_manager_.alpha_s_min(b, t); s <= workspace_manager_.alpha_s_max(b, t); ++s) {
            ProbT no_emit = workspace_manager_.get_alpha(b, t - 1, s)
                            + workspace_manager_.act(b, t, s, blank_)
                            + workspace_manager_.get_denom(b, t, s);
            ProbT emit = workspace_manager_.get_alpha(b, t - 1, s - 1);
            if (s > 0) {
                emit += workspace_manager_.act(b, t, s - 1, workspace_manager_.label(b, s - 1))
                        + workspace_manager_.get_denom(b, t, s - 1);

            }
            workspace_manager_.set_alpha(b, t, s, rnnt_helper::log_sum_exp<ProbT>(emit, no_emit));
        }
    }

#ifdef DEBUG_KERNEL
    printf("cpu alphas (b = %d, T = %d, S = %d):\n", b, workspace_manager_.T(b), workspace_manager_.S(b));
    for (int s = workspace_manager_.S(b); s >= 0; --s) {
        for (int t = -1; t < workspace_manager_.T(b); ++t) {
            printf("%.2f ", workspace_manager_.get_alpha(b, t, s));
        }
        printf("\n");
    }
    printf("\n");
#endif

    ProbT loglikelihood = workspace_manager_.get_alpha(b, workspace_manager_.T(b) - 1, workspace_manager_.S(b));

    return loglikelihood;
}


template<typename ProbT>
ProbT
CpuRNNTComputer<ProbT>::compute_betas_and_grad_kernel(int b, ProbT *grad) {
    for (int t = workspace_manager_.T(b) - 1; t >= 0; --t) {
        for (int s = workspace_manager_.beta_s_min(b, t); s <= workspace_manager_.beta_s_max(b, t); ++s) {
            ProbT no_emit = workspace_manager_.get_beta(b, t + 1, s)
                            + workspace_manager_.act(b, t, s, blank_)
                            + workspace_manager_.get_denom(b, t, s);

            ProbT emit = workspace_manager_.get_beta(b, t + 1, s + 1);
            if (s < workspace_manager_.S(b)) {
                emit += workspace_manager_.act(b, t, s, workspace_manager_.label(b, s))
                        + workspace_manager_.get_denom(b, t, s);
            }
            workspace_manager_.set_beta(b, t, s, rnnt_helper::log_sum_exp<ProbT>(emit, no_emit));
        }
    }

#ifdef DEBUG_KERNEL
    printf("cpu betas (b = %d, T = %d, S = %d):\n", b, workspace_manager_.T(b), workspace_manager_.S(b));
    for (int s = workspace_manager_.S(b); s >= 0; --s) {
        for (int t = 0; t <= workspace_manager_.T(b); ++t) {
            printf("%.2f ", workspace_manager_.get_beta(b, t, s));
        }
        printf("\n");
    }
    printf("\n");
#endif

    ProbT loglikelihood = workspace_manager_.get_beta(b, 0, 0);

    // Gradients w.r.t. logits
    for (int t = 0; t < workspace_manager_.T(b); ++t) {
        for (int s = 0; s <= workspace_manager_.S(b); ++s) {
            for (int v = 0; v < workspace_manager_.V(); ++v) {
                ProbT g = std::exp(
                        workspace_manager_.act(b, t, s, v)
                        + workspace_manager_.get_denom(b, t, s)
                        - loglikelihood
                        + workspace_manager_.get_alpha(b, t - 1, s)
                        + workspace_manager_.get_beta(b, t, s)
                );
                if (v == blank_) {
                    g -= std::exp(
                            workspace_manager_.act(b, t, s, v)
                            + workspace_manager_.get_denom(b, t, s)
                            - loglikelihood
                            + workspace_manager_.get_alpha(b, t - 1, s)
                            + workspace_manager_.get_beta(b, t + 1, s)
                    );
                } else if (s < workspace_manager_.S(b) && v == workspace_manager_.label(b, s)) {
                    g -= std::exp(
                            workspace_manager_.act(b, t, s, v)
                            + workspace_manager_.get_denom(b, t, s)
                            - loglikelihood
                            + workspace_manager_.get_alpha(b, t - 1, s)
                            + workspace_manager_.get_beta(b, t + 1, s + 1)
                    );
                }
                grad[workspace_manager_.act_index(b, t, s, v)] = g;
            }
        }
    }

#ifdef DEBUG_KERNEL
    printf("cpu grads (b = %d)\n", b);
    for (int t = 0; t < workspace_manager_.T(b); ++t) {
        printf("t = %d\n  ", t);
        for (int s = 0; s <= workspace_manager_.S(b); ++s) {
            for (int v = 0; v < workspace_manager_.V(); ++v) {
                printf("%.2f ", grad[workspace_manager_.act_index(b, t, s, v)]);
            }
            printf("\n  ");
        }
        printf("\n");
    }
    printf("\n");
#endif

    return loglikelihood;
}

template<typename ProbT>
ProbT
CpuRNNTComputer<ProbT>::cost_and_grad_kernel(int b, ProbT *grad) {
    ProbT llForward = compute_alphas_kernel(b);
    ProbT llBackward = compute_betas_and_grad_kernel(b, grad);

    if (std::abs(llForward - llBackward) > 1e-1) {
        printf("WARNING: Forward backward likelihood mismatch: %f vs. %f\n", llForward, llBackward);
    }

    return -llForward;
}

template<typename ProbT>
RNNTStatus
CpuRNNTComputer<ProbT>::cost_and_grad(ProbT *costs, ProbT *grads) {
#ifdef DEBUG_TIME
    auto start = std::chrono::high_resolution_clock::now();
#endif
    setup_log_softmax_denom();
#ifdef DEBUG_TIME
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "DEBUG: log_softmax denom " << elapsed.count() * 1000 << " ms\n";
    start = std::chrono::high_resolution_clock::now();
#endif

#pragma omp parallel for default(none) shared(costs, grads)
    for (int b = 0; b < workspace_manager_.B(); ++b) {
        costs[b] = cost_and_grad_kernel(b, grads);
    }

#ifdef DEBUG_TIME
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "DEBUG: alphas, betas and grads " << elapsed.count() * 1000 << " ms\n";
#endif

    return RNNT_STATUS_SUCCESS;
}

template<typename ProbT>
RNNTStatus
CpuRNNTComputer<ProbT>::cost(ProbT *costs) {
#ifdef DEBUG_TIME
    auto start = std::chrono::high_resolution_clock::now();
#endif
    setup_log_softmax_denom();
#ifdef DEBUG_TIME
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "DEBUG: log_softmax denom" << elapsed.count() * 1000 << " ms\n";
    start = std::chrono::high_resolution_clock::now();
#endif

#pragma omp parallel for default(none) shared(costs)
    for (int b = 0; b < workspace_manager_.B(); ++b) {
        costs[b] = -compute_alphas_kernel(b);
    }

#ifdef DEBUG_TIME
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "DEBUG: alphas " << elapsed.count() * 1000 << " ms\n";
#endif

    return RNNT_STATUS_SUCCESS;
}

#endif //MONOTONIC_RNNT_CPU_RNNT_H