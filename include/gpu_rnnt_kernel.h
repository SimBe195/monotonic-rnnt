#ifndef MONOTONIC_RNNT_GPU_RNNT_KERNEL_H
#define MONOTONIC_RNNT_GPU_RNNT_KERNEL_H

#include "gpu_workspace_manager.h"

//template<typename Tp>
//__global__ void
//compute_alphas_kernel(const Tp *const acts, const Tp *const get_denom, Tp *alphas, Tp *alpha_sil, Tp *llForward,
//                      const int *const xlen, const int *const ylen,
//                      const int *const mlabels, const int *const silence_indices, const int *const num_sil_indices,
//                      const int minibatch, const int *const start_indices, const int *const sil_start_indices,
//                      const int *const act_start_indices, const int max_S, const int alphabet_size, const int blank_,
//                      const int silence_) {
//    // launch B blocks, each block has U threads
//    int b = blockIdx.x; // batch
//    int s = threadIdx.x; // label id, s
//    const int T = xlen[b];
//    const int S = ylen[b];
//    const int S_sil = num_sil_indices[b];
//    const int *labels = mlabels + b * max_S; // mb label start point
//    const int start_idx = start_indices[b];
//    const int sil_start_idx = sil_start_indices[b];
//    const int act_start_idx = act_start_indices[b];
//
//    int silence_indices_start = 0;
//    for (int b_ = 0; b_ < b; ++b_) {
//        silence_indices_start += num_sil_indices[b_];
//    }
//
//    int s_idx = 0;
//    while (s_idx < S_sil && silence_indices[silence_indices_start + s_idx] < s) {
//        ++s_idx;
//    }
//    bool s_is_sil_idx = s_idx < S_sil && silence_indices[silence_indices_start + s_idx] == s;
//
//    alphas += start_idx;
//    alpha_sil += sil_start_idx;
//
//    if (s == 0) {
//        alphas[alpha_idx(0, 0, S)] = 0;
//    }
//
//    if (s_is_sil_idx) {
//        if (s == 0) {
//            alpha_sil[s_idx] = 0;
//        } else {
//            alpha_sil[s_idx] = rnnt_helper::neg_inf<Tp>();
//        }
//    }
//
//    __syncthreads();
//    for (int t = 1; t <= T; ++t) {
//        if (t < s) {
//            if (s_is_sil_idx) {
//                alpha_sil[t * S_sil + s_idx] = rnnt_helper::neg_inf<Tp>();
//            }
//        } else {
//            bool no_emit_possible = t - 1 >= s;
//            if (s == 0) { // only no_emit possible
//                alphas[alpha_idx(t, 0, S)] = alphas[alpha_idx(t - 1, 0, S)] +
//                                             logp(get_denom, acts, act_start_idx, S, alphabet_size, b, t - 1, 0,
//                                                  blank_);
//            } else if (s <= S) {
//                Tp emit = alphas[alpha_idx(t - 1, s - 1, S)] +
//                          logp(get_denom, acts, act_start_idx, S, alphabet_size, b, t - 1, s - 1, labels[s - 1]);
//                if (no_emit_possible) {
//                    Tp no_emit = alphas[alpha_idx(t - 1, s, S)] +
//                                 logp(get_denom, acts, act_start_idx, S, alphabet_size, b, t - 1, s, blank_);
//                    alphas[alpha_idx(t, s, S)] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
//                } else {
//                    alphas[alpha_idx(t, s, S)] = emit;
//                }
//            }
//
//            if (s_is_sil_idx) {
//                if (s == 0) {
//                    alpha_sil[t * S_sil] = alpha_sil[(t - 1) * S_sil] +
//                                           logp(get_denom, acts, act_start_idx, S, alphabet_size, b, t - 1, 0, blank_);
//                } else {
//                    Tp emit = alphas[alpha_idx(t - 1, s - 1, S)] +
//                              logp(get_denom, acts, act_start_idx, S, alphabet_size, b, t - 1, s - 1, labels[s - 1]);
//                    if (no_emit_possible) {
//                        Tp no_emit = alpha_sil[(t - 1) * S_sil + s_idx] +
//                                     logp(get_denom, acts, act_start_idx, S, alphabet_size, b, t - 1, s, blank_);
//                        alpha_sil[t * S_sil + s_idx] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
//                    } else {
//                        alpha_sil[t * S_sil + s_idx] = emit;
//                    }
//                }
//                Tp sil_emit = alpha_sil[(t - 1) * S_sil + s_idx] +
//                              logp(get_denom, acts, act_start_idx, S, alphabet_size, b, t - 1, s, silence_);
//                alphas[alpha_idx(t, s, S)] = rnnt_helper::log_sum_exp<Tp>(alphas[alpha_idx(t, s, S)], sil_emit);
//            }
//        }
//        __syncthreads();
//    }
//
//    if (s == S) { // s == S specifically is not important, but this only has to be done once.
//        llForward[b] = alphas[alpha_idx(T, S, S)];
//    }
//}

template<typename Tp>
__global__ void
compute_alphas_kernel_naive(const Tp *const acts, const Tp *const denom, Tp *alphas, Tp *ll_forward, const int *const T,
                            const int *const S, const int V, const int *const labels,
                            const int *const var_start_offsets, const int *const denom_start_indices, const int S_max,
                            const int blank_) {
    int b = static_cast<int>(threadIdx.x);
    const int T_b = T[b];
    const int S_b = S[b];
    const int *labels_b = labels + b * S_max;
    const int var_start_idx_b = var_start_offsets[b];
    const int denom_start_idx_b = denom_start_indices[b];
    const Tp *const acts_b = acts + denom_start_idx_b * V;
    const Tp *const denom_b = denom + denom_start_idx_b;
    Tp *alphas_b = alphas + var_start_idx_b;

    for (int t = 0; t < T_b; ++t) {
        for (int s = alpha_s_min(t, T_b, S_b); s <= alpha_s_max(t, T_b, S_b); ++s) {
            Tp no_emit = alpha(alphas_b, t - 1, s, T_b, S_b) + log_p(acts_b, denom_b, t, s, blank_, S_b, V);
            Tp emit = alpha(alphas_b, t - 1, s - 1, T_b, S_b);
            if (s > 0) {
                emit += log_p(acts_b, denom_b, t, s - 1, labels_b[s - 1], S_b, V);
            }

            alphas_b[alpha_idx(t, s, T_b, S_b)] = rnnt_helper::log_sum_exp(no_emit, emit);
        }
    }

    ll_forward[b] = alpha(alphas_b, T_b - 1, S_b, T_b, S_b);
}


//template<typename Tp>
//__global__ void
//compute_betas_kernel(const Tp *const acts, const Tp *const get_denom, Tp *betas, Tp *beta_sil, Tp *llBackward,
//                     const int *const xlen, const int *const ylen,
//                     const int *const mlabels, const int *const silence_indices, const int *const num_sil_indices,
//                     const int minibatch, const int *const start_indices, const int *const sil_start_indices,
//                     const int *const act_start_indices, const int max_S, const int alphabet_size, const int blank_,
//                     const int silence_) {
//    int b = blockIdx.x; // batch
//    int s = threadIdx.x; // label id, s
//    const int T = xlen[b];
//    const int S = ylen[b];
//    const int S_sil = num_sil_indices[b];
//    const int *labels = mlabels + b * max_S;
//    const int start_idx = start_indices[b];
//    const int sil_start_idx = sil_start_indices[b];
//    const int act_start_idx = act_start_indices[b];
//
//    int silence_indices_start = 0;
//    for (int b_ = 0; b_ < b; ++b_) {
//        silence_indices_start += num_sil_indices[b_];
//    }
//
//    int s_idx = 0;
//    while (s_idx < S_sil && silence_indices[silence_indices_start + s_idx] < s) {
//        ++s_idx;
//    }
//    bool s_is_sil_idx = s_idx < S_sil && silence_indices[silence_indices_start + s_idx] == s;
//
//    betas += start_idx;
//    beta_sil += sil_start_idx;
//
//    if (s == S) {
//        betas[beta_idx(T, S, T, S)] = 0;
//    }
//
//    if (s_is_sil_idx) {
//        if (s == S) {
//            beta_sil[T * S_sil + s_idx] = 0;
//        } else {
//            beta_sil[T * S_sil + s_idx] = rnnt_helper::neg_inf<Tp>();
//        }
//    }
//
//    __syncthreads();
//    for (int t = T - 1; t >= 0; --t) {
//        if (S - s > T - t) {
//            if (s_is_sil_idx) {
//                beta_sil[t * S_sil + s_idx] = rnnt_helper::neg_inf<Tp>();
//            }
//        } else {
//            bool no_emit_possible = S - s <= T - (t + 1);
//            if (s == S) {
//                betas[beta_idx(t, S, T, S)] = betas[beta_idx(t + 1, S, T, S)] +
//                                              logp(get_denom, acts, act_start_idx, S, alphabet_size, b, t, S, blank_);
//            } else if (s < S) {
//                Tp emit = betas[beta_idx(t + 1, s + 1, T, S)] +
//                          logp(get_denom, acts, act_start_idx, S, alphabet_size, b, t, s, labels[s]);
//                if (no_emit_possible) {
//                    Tp no_emit = betas[beta_idx(t + 1, s, T, S)] +
//                                 logp(get_denom, acts, act_start_idx, S, alphabet_size, b, t, s, blank_);
//                    betas[beta_idx(t, s, T, S)] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
//                } else {
//                    betas[beta_idx(t, s, T, S)] = emit;
//                }
//            }
//
//            if (s_is_sil_idx) {
//                if (s == S) {
//                    beta_sil[t * S_sil + s_idx] = beta_sil[(t + 1) * S_sil + s_idx] +
//                                                  logp(get_denom, acts, act_start_idx, S, alphabet_size, b, t, S,
//                                                       blank_);
//                } else {
//                    Tp emit = betas[beta_idx(t + 1, s + 1, T, S)] +
//                              logp(get_denom, acts, act_start_idx, S, alphabet_size, b, t, s, labels[s]);
//                    if (no_emit_possible) {
//                        Tp no_emit = beta_sil[(t + 1) * S_sil + s_idx] +
//                                     logp(get_denom, acts, act_start_idx, S, alphabet_size, b, t, s, blank_);
//                        beta_sil[t * S_sil + s_idx] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
//                    } else {
//                        beta_sil[t * S_sil + s_idx] = emit;
//                    }
//                }
//                Tp sil_emit = beta_sil[(t + 1) * S_sil + s_idx] +
//                              logp(get_denom, acts, act_start_idx, S, alphabet_size, b, t, s, silence_);
//                betas[beta_idx(t, s, T, S)] = rnnt_helper::log_sum_exp<Tp>(betas[beta_idx(t, s, T, S)], sil_emit);
//            }
//        }
//        __syncthreads();
//    }
//
//    if (s == 0) { // s == 0 specifically is not important, but this only has to be done once.
//        llBackward[b] = betas[beta_idx(0, 0, T, S)];
//    }
//}

template<typename Tp>
__global__ void
compute_betas_kernel_naive(const Tp *const acts, const Tp *const denom, Tp *betas, Tp *ll_backward, const int *const T,
                           const int *const S, const int V, const int *const labels, const int *const var_start_offsets,
                           const int *const denom_start_indices, const int S_max, const int blank_) {
    int b = static_cast<int>(threadIdx.x);
    const int T_b = T[b];
    const int S_b = S[b];
    const int *labels_b = labels + b * S_max;
    const int var_start_offset_b = var_start_offsets[b];
    const int denom_start_idx_b = denom_start_indices[b];
    const Tp *const acts_b = acts + denom_start_idx_b * V;
    const Tp *const denom_b = denom + denom_start_idx_b;
    Tp *betas_b = betas + var_start_offset_b;

    for (int t = T_b - 1; t >= 0; --t) {
        for (int s = beta_s_min(t, T_b, S_b); s <= beta_s_max(t, T_b, S_b); ++s) {
            Tp no_emit = beta(betas_b, t, s, T_b, S_b) + log_p(acts_b, denom_b, t, s, blank_, S_b, V);
            Tp emit = beta(betas_b, t + 1, s + 1, T_b, S_b);
            if (s < S_b) {
                emit += log_p(acts_b, denom_b, t, s, labels_b[s], S_b, V);
            }
            betas_b[beta_idx(t, s, T_b, S_b)] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
        }
    }

    ll_backward[b] = betas[0];
}

template<int NT, typename Tp>
__global__ void
compute_grad_kernel(Tp *grads, const Tp *const acts, const Tp *const denom, const Tp *alphas, const Tp *betas,
                    const Tp *const logll, const int B, const int *const T, const int *const S, const int *const labels,
                    const int *const var_start_offsets, const int *const denom_start_indices, const int S_max,
                    const int V, const int blank_) {
    int v = static_cast<int>(threadIdx.x);
    int bts = static_cast<int>(blockIdx.x); // b, t, s packed

    int b = 0;
    while (b < B - 1 && denom_start_indices[b + 1] <= bts) {
        ++b;
    }

    const int T_b = T[b];
    const int S_b = S[b];
    const int *labels_b = labels + b * S_max;
    const Tp *alphas_b = alphas + var_start_offsets[b];
    const Tp *betas_b = betas + var_start_offsets[b];

    int ts = bts - denom_start_indices[b];
    int t = ts / (S_b + 1);
    int s = ts % (S_b + 1);

    if (t < s || T_b - t < S_b - s) {
        for (; v < V; v += NT) {
            grads[bts * V + v] = 0;
        }
        return;
    }

    for (; v < V; v += NT) {
        Tp logpk = denom[bts] + acts[bts * V + v];
        Tp grad = exp(
                logpk
                - logll[b]
                + alpha(alphas_b, t - 1, s, T_b, S_b)
                + beta(betas_b, t, s, T_b, S_b)
        );

        if (v == blank_) {
            grad -= exp(
                    logpk
                    - logll[b]
                    + alpha(alphas_b, t - 1, s, T_b, S_b)
                    + beta(betas_b, t + 1, s, T_b, S_b)
            );
        } else if (s < S_b && v == labels_b[s]) {
            grad -= exp(
                    logpk
                    - logll[b]
                    + alpha(alphas_b, t - 1, s, T_b, S_b)
                    + beta(betas_b, t + 1, s, T_b, S_b)
            );
        }

        grads[bts * V + v] = grad;
    }
}

#endif //MONOTONIC_RNNT_GPU_RNNT_KERNEL_H
