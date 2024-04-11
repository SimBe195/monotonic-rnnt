#ifndef MONOTONIC_RNNT_GPU_RNNT_KERNEL_H
#define MONOTONIC_RNNT_GPU_RNNT_KERNEL_H

#include "gpu_workspace_manager.h"

HOSTDEVICE int alpha_idx(const int t, const int s, const int T, const int S) {
    // see cpu workspace manager for more detailed explanation. This is broken
    // down for efficiency.
    int left_portion = t < S - 1 ? t : S - 1;
    int mid_portion = t < S - 1 ? 0 : (t < T - S ? t + 1 - S : T + 1 - 2 * S);
    int right_portion = t < T - S ? 0 : t - T + S;

    int left_sum = (left_portion + 1) * (left_portion + 2) / 2 - 1;
    int mid_sum = mid_portion * (S + 1);
    int right_sum = right_portion * (S + 1) - right_portion * (right_portion + 1) / 2;

    int height_offset = t < T - S ? 0 : -right_portion - 1;

    return left_sum + mid_sum + right_sum + height_offset + s;
}

inline HOSTDEVICE int beta_idx(const int t, const int s, const int T, const int S) {
    return alpha_idx(T - 1 - t, S - s, T, S);
}

template <typename dtype>
HOSTDEVICE dtype alpha(dtype *alphas, const int t, const int s, const int T, const int S) {
    // Note: t = -1 and s = -1 are allowed to be used as virtual starts
    // They correspond to constants
    if (s == -1) {
        return rnnt_helper::neg_inf<dtype>();
    }

    if (t == -1) {
        return s == 0 ? 0 : rnnt_helper::neg_inf<dtype>();
    }

    if (s > t + 1 || S - s > T - 1 - t) {
        return rnnt_helper::neg_inf<dtype>();
    }

    return alphas[alpha_idx(t, s, T, S)];
}

template <typename dtype>
HOSTDEVICE dtype beta(dtype *betas, const int t, const int s, const int T, const int S) {
    // Note: t = T and s = S+1 are allowed to be used as virtual starts
    // They correspond to constants
    if (s == S + 1) {
        return rnnt_helper::neg_inf<dtype>();
    }

    if (t == T) {
        return s == S ? 0 : rnnt_helper::neg_inf<dtype>();
    }

    if (s > t || S - s - 1 > T - 1 - t) {
        return rnnt_helper::neg_inf<dtype>();
    }

    return betas[beta_idx(t, s, T, S)];
}

inline HOSTDEVICE int alpha_s_min(const int t, const int T, const int S) { return t < T - 1 - S ? 0 : t - (T - 1 - S); }

inline HOSTDEVICE int alpha_s_max(const int t, const int S) { return t < S ? t + 1 : S; }

inline HOSTDEVICE int beta_s_min(const int t, const int T, const int S) { return t < T - S ? 0 : t - (T - S); }

inline HOSTDEVICE int beta_s_max(const int t, const int S) { return t < S ? t : S; }

inline HOSTDEVICE int denom_idx(const int t, const int s, const int S) { return t * (S + 1) + s; }

inline HOSTDEVICE int act_idx(const int t, const int s, const int v, const int S, const int V) {
    return (denom_idx(t, s, S)) * V + v;
}

template <typename dtype>
inline HOSTDEVICE dtype log_p(const dtype *const acts, const dtype *const denom, const int t, const int s, const int v,
                              const int S, const int V) {
    return acts[act_idx(t, s, v, S, V)] + denom[denom_idx(t, s, S)];
}

template <typename Tp>
__global__ void compute_alphas_kernel_naive(const Tp *const acts, const Tp *const denom, Tp *alphas, Tp *ll_forward,
                                            const int *const T, const int *const S, const int *const V,
                                            const int *const labels, const int *const var_start_offsets,
                                            const int *const denom_start_indices, const int *const S_max,
                                            const int blank_) {
    int b = static_cast<int>(threadIdx.x);
    const int T_b = T[b];
    const int S_b = S[b];
    const int *labels_b = labels + b * *S_max;
    const int var_start_idx_b = var_start_offsets[b];
    const int denom_start_idx_b = denom_start_indices[b];
    const Tp *const acts_b = acts + denom_start_idx_b * *V;
    const Tp *const denom_b = denom + denom_start_idx_b;
    Tp *alphas_b = alphas + var_start_idx_b;

    for (int t = 0; t < T_b; ++t) {
        for (int s = alpha_s_min(t, T_b, S_b); s <= alpha_s_max(t, S_b); ++s) {
            Tp no_emit = alpha(alphas_b, t - 1, s, T_b, S_b) + log_p(acts_b, denom_b, t, s, blank_, S_b, *V);
            Tp emit = alpha(alphas_b, t - 1, s - 1, T_b, S_b);
            if (s > 0) {
                emit += log_p(acts_b, denom_b, t, s - 1, labels_b[s - 1], S_b, *V);
            }

            alphas_b[alpha_idx(t, s, T_b, S_b)] = rnnt_helper::log_sum_exp(no_emit, emit);
        }
    }

    ll_forward[b] = alpha(alphas_b, T_b - 1, S_b, T_b, S_b);
}

template <typename Tp>
__global__ void compute_alphas_kernel(const Tp *const acts, const Tp *const denom, Tp *alphas, Tp *ll_forward,
                                      const int *const T, const int *const S, const int *const V,
                                      const int *const labels, const int *const var_start_offsets,
                                      const int *const denom_start_indices, const int *const S_max, const int blank_) {
    // launch B blocks, each block has S+1 threads
    int b = static_cast<int>(blockIdx.x);
    int s = static_cast<int>(threadIdx.x);
    const int T_b = T[b];
    const int S_b = S[b];
    const int *labels_b = labels + b * *S_max;
    const int var_start_idx_b = var_start_offsets[b];
    const int denom_start_idx_b = denom_start_indices[b];
    const Tp *const acts_b = acts + denom_start_idx_b * *V;
    const Tp *const denom_b = denom + denom_start_idx_b;
    Tp *alphas_b = alphas + var_start_idx_b;

    __syncthreads();
    for (int t = 0; t < T_b; ++t) {
        if (s >= alpha_s_min(t, T_b, S_b) && s <= alpha_s_max(t, S_b)) {
            Tp no_emit = alpha(alphas_b, t - 1, s, T_b, S_b) + log_p(acts_b, denom_b, t, s, blank_, S_b, *V);
            Tp emit = alpha(alphas_b, t - 1, s - 1, T_b, S_b);
            if (s > 0) {
                emit += log_p(acts_b, denom_b, t, s - 1, labels_b[s - 1], S_b, *V);
            }

            alphas_b[alpha_idx(t, s, T_b, S_b)] = rnnt_helper::log_sum_exp(no_emit, emit);
        }
        __syncthreads();
    }

    if (s == S_b) {  // s == S specifically is not important, but this only has
                     // to be done once.
        ll_forward[b] = alpha(alphas_b, T_b - 1, S_b, T_b, S_b);
    }
}

template <typename Tp>
__global__ void compute_betas_kernel_naive(const Tp *const acts, const Tp *const denom, Tp *betas, Tp *ll_backward,
                                           const int *const T, const int *const S, const int *const V,
                                           const int *const labels, const int *const var_start_offsets,
                                           const int *const denom_start_indices, const int *const S_max,
                                           const int blank_) {
    int b = static_cast<int>(threadIdx.x);
    const int T_b = T[b];
    const int S_b = S[b];
    const int *labels_b = labels + b * *S_max;
    const int var_start_offset_b = var_start_offsets[b];
    const int denom_start_idx_b = denom_start_indices[b];
    const Tp *const acts_b = acts + denom_start_idx_b * *V;
    const Tp *const denom_b = denom + denom_start_idx_b;
    Tp *betas_b = betas + var_start_offset_b;

    for (int t = T_b - 1; t >= 0; --t) {
        for (int s = beta_s_min(t, T_b, S_b); s <= beta_s_max(t, S_b); ++s) {
            Tp no_emit = beta(betas_b, t + 1, s, T_b, S_b) + log_p(acts_b, denom_b, t, s, blank_, S_b, *V);
            Tp emit = beta(betas_b, t + 1, s + 1, T_b, S_b);
            if (s < S_b) {
                emit += log_p(acts_b, denom_b, t, s, labels_b[s], S_b, *V);
            }
            betas_b[beta_idx(t, s, T_b, S_b)] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
        }
    }

    ll_backward[b] = beta(betas_b, 0, 0, T_b, S_b);
}

template <typename Tp>
__global__ void compute_betas_kernel(const Tp *const acts, const Tp *const denom, Tp *betas, Tp *ll_backward,
                                     const int *const T, const int *const S, const int *const V,
                                     const int *const labels, const int *const var_start_offsets,
                                     const int *const denom_start_indices, const int *const S_max, const int blank_) {
    // launch B blocks, each block has S+1 threads
    int b = static_cast<int>(blockIdx.x);
    int s = static_cast<int>(threadIdx.x);
    const int T_b = T[b];
    const int S_b = S[b];
    const int *labels_b = labels + b * *S_max;
    const int var_start_idx_b = var_start_offsets[b];
    const int denom_start_idx_b = denom_start_indices[b];
    const Tp *const acts_b = acts + denom_start_idx_b * *V;
    const Tp *const denom_b = denom + denom_start_idx_b;
    Tp *betas_b = betas + var_start_idx_b;

    __syncthreads();
    for (int t = T_b - 1; t >= 0; --t) {
        if (s >= beta_s_min(t, T_b, S_b) && s <= beta_s_max(t, S_b)) {
            Tp no_emit = beta(betas_b, t + 1, s, T_b, S_b) + log_p(acts_b, denom_b, t, s, blank_, S_b, *V);
            Tp emit = beta(betas_b, t + 1, s + 1, T_b, S_b);
            if (s < S_b) {
                emit += log_p(acts_b, denom_b, t, s, labels_b[s], S_b, *V);
            }

            betas_b[beta_idx(t, s, T_b, S_b)] = rnnt_helper::log_sum_exp(no_emit, emit);
        }

        __syncthreads();
    }

    if (s == 0) {  // s == 0 specifically is not important, but this only has to
                   // be done once.
        ll_backward[b] = beta(betas_b, 0, 0, T_b, S_b);
    }
}

template <int NT, typename Tp>
__global__ void compute_grad_kernel(Tp *grads, const Tp *const acts, const Tp *const denom, const Tp *const alphas,
                                    const Tp *const betas, const Tp *const logll, const int *const B,
                                    const int *const T, const int *const S, const int *const labels,
                                    const int *const var_start_offsets, const int *const denom_start_indices,
                                    const int *const S_max, const int *const V, const int blank_) {
    int v = static_cast<int>(threadIdx.x);
    int bts = static_cast<int>(blockIdx.x);  // b, t, s packed

    int b = 0;
    while (b < *B - 1 && denom_start_indices[b + 1] <= bts) {
        ++b;
    }

    const int T_b = T[b];
    const int S_b = S[b];
    const int *labels_b = labels + b * *S_max;
    const Tp *alphas_b = alphas + var_start_offsets[b];
    const Tp *betas_b = betas + var_start_offsets[b];

    int ts = bts - denom_start_indices[b];
    int t = ts / (S_b + 1);
    int s = ts % (S_b + 1);

    if (t < s || T_b - t < S_b - s) {
        for (; v < *V; v += NT) {
            grads[bts * *V + v] = 0;
        }
        return;
    }

    for (; v < *V; v += NT) {
        Tp logpk = denom[bts] + acts[bts * *V + v];
        Tp grad = exp(logpk - logll[b] + alpha(alphas_b, t - 1, s, T_b, S_b) + beta(betas_b, t, s, T_b, S_b));

        if (v == blank_) {
            grad -= exp(logpk - logll[b] + alpha(alphas_b, t - 1, s, T_b, S_b) + beta(betas_b, t + 1, s, T_b, S_b));
        } else if (s < S_b && v == labels_b[s]) {
            grad -= exp(logpk - logll[b] + alpha(alphas_b, t - 1, s, T_b, S_b) + beta(betas_b, t + 1, s + 1, T_b, S_b));
        }

        grads[bts * *V + v] = grad;
    }
}

#endif  // MONOTONIC_RNNT_GPU_RNNT_KERNEL_H
