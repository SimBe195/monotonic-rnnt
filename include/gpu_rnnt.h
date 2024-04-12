#ifndef MONOTONIC_RNNT_GPU_RNNT_H
#define MONOTONIC_RNNT_GPU_RNNT_H

#if defined(DEBUG_TIME) or defined(DEBUG_KERNEL)
#include <stdio.h>
#endif

#ifdef DEBUG_TIME
#include <chrono>
#endif

#include "gpu_rnnt_kernel.h"
#include "gpu_workspace_manager.h"
#include "reduce.h"

template <typename ProbT>
class GpuRNNTComputer {
   public:
    // Noncopyable
    GpuRNNTComputer(GpuRNNTWorkspaceManager<ProbT> &workspace_manager, int blank, CUstream stream)
        : workspace_manager_(workspace_manager), blank_(blank), stream_(stream) {}

    GpuRNNTComputer(const GpuRNNTComputer &) = delete;

    GpuRNNTComputer &operator=(const GpuRNNTComputer &) = delete;

    RNNTStatus cost_and_grad(ProbT *costs, ProbT *grads) {
        std::vector<int> lengths_test(5);
        int B = workspace_manager_.B_host();
        auto T = workspace_manager_.T_host(stream_);
        auto S = workspace_manager_.S_host(stream_);
        int V = workspace_manager_.V_host();
        int S_max = workspace_manager_.S_max_host(stream_);

        bool training = (grads != nullptr);

        if (training) {
            // zero grads
            cudaMemsetAsync(grads, 0, sizeof(ProbT) * workspace_manager_.num_denoms(stream_) * V, stream_);
        }

        // denom

#ifdef DEBUG_TIME
        auto start = std::chrono::high_resolution_clock::now();
#endif
        setup_log_softmax_denom();
#ifdef DEBUG_TIME
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        printf("DEBUG: log_softmax denom %.2f ms\n", elapsed.count() * 1000);
        start = std::chrono::high_resolution_clock::now();
#endif

#ifdef DEBUG_KERNEL
        auto cpu_acts = workspace_manager_.acts_host(stream_);
        auto cpu_denoms = workspace_manager_.denom_host(stream_);
        printf("gpu acts and denoms\n");
        int denom_idx = 0;
        for (int b = 0; b < B; b++) {
            printf("b = %d\n", b);
            for (int t = 0; t < T[b]; t++) {
                printf("  t = %d\n", t);
                for (int s = 0; s <= S[b]; s++) {
                    printf("    s = %d\n      ", s);
                    for (int v = 0; v < V; v++) {
                        printf("%.4f ", cpu_acts[denom_idx * V + v]);
                    }
                    printf("=> %.4f;\n", cpu_denoms[denom_idx]);
                    denom_idx += 1;
                }
                printf("\n");
            }
            printf("\n");
        }
#endif

        // alphas

#ifdef USE_NAIVE_KERNEL
        compute_alphas_kernel_naive<ProbT><<<1, B, 0, stream_>>>(
            workspace_manager_.acts, workspace_manager_.denom, workspace_manager_.alphas, workspace_manager_.ll_forward,
            workspace_manager_.T, workspace_manager_.S, workspace_manager_.V, workspace_manager_.labels,
            workspace_manager_.var_start_offsets, workspace_manager_.denom_start_indices, workspace_manager_.S_max,
            blank_);
#else
        compute_alphas_kernel<ProbT><<<B, S_max + 1, 0, stream_>>>(
            workspace_manager_.acts, workspace_manager_.denom, workspace_manager_.alphas, workspace_manager_.ll_forward,
            workspace_manager_.T, workspace_manager_.S, workspace_manager_.V, workspace_manager_.labels,
            workspace_manager_.var_start_offsets, workspace_manager_.denom_start_indices, workspace_manager_.S_max,
            blank_);
#endif
#ifdef DEBUG_TIME
        cudaStreamSynchronize(stream_);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        printf("DEBUG: compute_alphas_kernel %.2f ms\n", elapsed.count() * 1000);
#endif
#ifdef DEBUG_KERNEL
        auto alphas = workspace_manager_.alphas_host(stream_);
        auto var_start_offsets = workspace_manager_.var_start_offsets_host(stream_);
        for (int b = 0; b < B; b++) {
            printf("gpu alphas (b = %d, T = %d, S = %d):\n", b, T[b], S[b]);
            float *alphas_b = alphas.data() + var_start_offsets[b];
            for (int s = S[b]; s >= 0; --s) {
                for (int t = -1; t < T[b]; ++t) {
                    printf("%.2f ", alpha(alphas_b, t, s, T[b], S[b]));
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("forward likelihood\n");
        auto ll_forward = workspace_manager_.ll_forward_host(stream_);
        for (int b = 0; b < B; ++b) {
            printf("%.2f ", ll_forward[b]);
        }
        printf("\n\n");
#endif
        if (training) {
            // betas
#ifdef DEBUG_TIME
            start = std::chrono::high_resolution_clock::now();
#endif
            cudaMemcpy(lengths_test.data(), workspace_manager_.T, sizeof(int) * lengths_test.size(), cudaMemcpyDeviceToHost);
            for (int & l : lengths_test) {
                l = 0;
            }
#ifdef USE_NAIVE_KERNEL
            compute_betas_kernel_naive<ProbT><<<1, B, 0, stream_>>>(
                workspace_manager_.acts, workspace_manager_.denom, workspace_manager_.betas,
                workspace_manager_.ll_backward, workspace_manager_.T, workspace_manager_.S, workspace_manager_.V,
                workspace_manager_.labels, workspace_manager_.var_start_offsets, workspace_manager_.denom_start_indices,
                workspace_manager_.S_max, blank_);
#else
            compute_betas_kernel<ProbT><<<B, S_max + 1, 0, stream_>>>(
                workspace_manager_.acts, workspace_manager_.denom, workspace_manager_.betas,
                workspace_manager_.ll_backward, workspace_manager_.T, workspace_manager_.S, workspace_manager_.V,
                workspace_manager_.labels, workspace_manager_.var_start_offsets, workspace_manager_.denom_start_indices,
                workspace_manager_.S_max, blank_);
#endif
            cudaMemcpy(lengths_test.data(), workspace_manager_.T, sizeof(int) * lengths_test.size(), cudaMemcpyDeviceToHost);
            for (int & l : lengths_test) {
                l = 0;
            }
#ifdef DEBUG_TIME
            cudaStreamSynchronize(stream_);
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            printf("DEBUG: compute_betas_kernel %.2f ms\n", elapsed.count() * 1000);
#endif
#ifdef DEBUG_KERNEL
            auto betas = workspace_manager_.betas_host(stream_);
            for (int b = 0; b < B; b++) {
                printf("gpu betas (b = %d, T = %d, S = %d):\n", b, T[b], S[b]);
                float *betas_b = betas.data() + var_start_offsets[b];
                for (int s = S[b]; s >= 0; --s) {
                    for (int t = 0; t <= T[b]; ++t) {
                        printf("%.2f ", beta(betas_b, t, s, T[b], S[b]));
                    }
                    printf("\n");
                }
                printf("\n");
            }
            printf("backward likelihood\n");
            auto ll_backward = workspace_manager_.ll_backward_host(stream_);
            for (int b = 0; b < B; ++b) {
                printf("%.2f ", ll_backward[b]);
            }
            printf("\n\n");
#endif

            // gradient
#ifdef DEBUG_TIME
            start = std::chrono::high_resolution_clock::now();
#endif
            compute_grad_kernel<128, ProbT><<<workspace_manager_.num_denoms(stream_), 128, 0, stream_>>>(
                grads, workspace_manager_.acts, workspace_manager_.denom, workspace_manager_.alphas,
                workspace_manager_.betas, workspace_manager_.ll_forward, workspace_manager_.B, workspace_manager_.T,
                workspace_manager_.S, workspace_manager_.labels, workspace_manager_.var_start_offsets,
                workspace_manager_.denom_start_indices, workspace_manager_.S_max, workspace_manager_.V, blank_);
#ifdef DEBUG_TIME
            cudaStreamSynchronize(stream_);
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            printf("DEBUG: compute_grad_kernel %.2f ms\n", elapsed.count() * 1000);
#endif
#ifdef DEBUG_KERNEL
            std::vector<ProbT> cpu_grads(workspace_manager_.num_denoms(stream_) * V);
            cudaMemcpyAsync(cpu_grads.data(), grads, sizeof(ProbT) * cpu_grads.size(), cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_);

            printf("gpu grads\n");
            int grad_idx = 0;
            for (int b = 0; b < B; ++b) {
                printf("b = %d\n", b);
                for (int t = 0; t < T[b]; ++t) {
                    printf("  t = %d\n", t);
                    for (int s = 0; s <= S[b]; ++s) {
                        printf("    s = %d\n      ", s);
                        for (int v = 0; v < V; ++v) {
                            printf("%.4f ", cpu_grads[grad_idx]);
                            grad_idx += 1;
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
                printf("\n");
            }
#endif
        }

        // cost
        cudaMemcpyAsync(costs, workspace_manager_.ll_forward, sizeof(ProbT) * B, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        for (int b = 0; b < B; ++b) {
            costs[b] = -costs[b];
        }
        return RNNT_STATUS_SUCCESS;
    }
    RNNTStatus cost(ProbT *costs) { return cost_and_grad(costs, nullptr); }

   private:
    GpuRNNTWorkspaceManager<ProbT> &workspace_manager_;
    int blank_;
    CUstream stream_;

    void setup_log_softmax_denom() {
        const int num_denoms = workspace_manager_.num_denoms(stream_);
        const int V = workspace_manager_.V_host();

        // trans_acts + pred_acts -> log_softmax denominator
        reduce_max(workspace_manager_.acts, workspace_manager_.denom, V, num_denoms, false, stream_);
        reduce_exp(workspace_manager_.acts, workspace_manager_.denom, V, num_denoms, true, stream_);
    }
};

#endif  // MONOTONIC_RNNT_GPU_RNNT_H
