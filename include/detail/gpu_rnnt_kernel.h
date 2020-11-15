#pragma once

#include "rnnt_helper.h"

template<typename T>
inline __device__ T logp(const T* const denom, const T* const acts, const int start_idx, const int U, const int alphabet_size, int mb, int t, int u, int v) {
    const int col = start_idx + t*U + u;
    return denom[col] + acts[col * alphabet_size + v];
}

template<typename Tp>
__global__ void compute_alphas_kernel(const Tp* const acts, const Tp* const denom, Tp* alphas, Tp* llForward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int* const start_indices, const int maxU, const int alphabet_size, const int blank_) {
    // launch B blocks, each block has U threads
    int b = blockIdx.x; // batch
    int u = threadIdx.x; // label id, u
    const int T = xlen[b];
    const int U = ylen[b] + 1;
    const int* labels = mlabels + b * (maxU - 1); // mb label start point
    const int start_idx = start_indices[b];
    alphas += start_idx;
    if (u == 0) { 
        alphas[0] = 0;
    } else if (u < U) {
        alphas[u] = rnnt_helper::neg_inf<Tp>();
    }

    __syncthreads();
    for (int t = 1; t < T; ++t) {
        if (u == 0) { // only no_emit possible
            alphas[t * U] = alphas[(t-1) * U] + logp(denom, acts, start_idx, U, alphabet_size, b, t-1, 0, blank_);
        } else if (u < U) {
            Tp no_emit = alphas[(t-1) * U + u] + logp(denom, acts, start_idx, U, alphabet_size, b, t-1, u, blank_);
            Tp emit = alphas[(t-1) * U + u-1] + logp(denom, acts, start_idx, U, alphabet_size, b, t-1, u-1, labels[u-1]);
            alphas[t * U + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
        }
        __syncthreads();
    }

    if (u == 0) { // u == 0 specifically is not important, but this only has to be done once.
        llForward[b] = alphas[T * U - 1] + logp(denom, acts, start_idx, U, alphabet_size, b, T-1, U-1, blank_);
    }
}

template<typename Tp>
__global__ void compute_alphas_kernel_naive(const Tp* const acts, const Tp* const denom, Tp* alphas, Tp* llForward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int* const start_indices, const int maxU, const int alphabet_size, const int blank_) {
    int tid = threadIdx.x; // mb
    const int T = xlen[tid];
    const int U = ylen[tid] + 1;
    const int* labels = mlabels + tid * (maxU - 1); // mb label start point
    const int start_idx = start_indices[tid];
    alphas += start_idx;
    alphas[0] = 0;

    for (int u = 1; u < U; ++u) {
        alphas[u] = rnnt_helper::neg_inf<Tp>();
    }

    for (int t = 1; t < T; ++t) {
        alphas[t * U] = alphas[(t-1) * U] + logp(denom, acts, start_idx, U, alphabet_size, tid, t-1, 0, blank_); // u = 0
        for (int u = 1; u < U; ++u) {
            Tp no_emit = alphas[(t-1) * U + u] + logp(denom, acts, start_idx, U, alphabet_size, tid, t-1, u, blank_);
            Tp emit = alphas[(t-1) * U + u-1] + logp(denom, acts, start_idx, U, alphabet_size, tid, t-1, u-1, labels[u-1]);
            alphas[t * U + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
        }
    }

    llForward[tid] = alphas[T * U - 1] + logp(denom, acts, start_idx, U, alphabet_size, tid, T-1, U-1, blank_);
}


template<typename Tp>
__global__ void compute_betas_kernel(const Tp* const acts, const Tp* const denom, Tp* betas, Tp* llBackward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int* const start_indices, const int maxU, const int alphabet_size, const int blank_) {
    int b = blockIdx.x; // batch
    int u = threadIdx.x; // label id, u
    const int T = xlen[b];
    const int U = ylen[b] + 1;
    const int* labels = mlabels + b * (maxU - 1);
    const int start_idx = start_indices[b];
    betas += start_idx;

    if (u == U-1) {
        betas[T*U - 1] = logp(denom, acts, start_idx, U, alphabet_size, b, T-1, U-1, blank_);
    } else if (u < U) {
        betas[(T-1) * U + u] = rnnt_helper::neg_inf<Tp>();
    }

    __syncthreads();
    for (int t = T-2; t >= 0; --t) {
        if (u == U-1) {
            betas[t * U + U-1] = betas[(t+1) * U + U-1] + logp(denom, acts, start_idx, U, alphabet_size, b, t, U-1, blank_);
        } else if (u < U-1) {
            Tp no_emit = betas[(t+1) * U + u] + logp(denom, acts, start_idx, U, alphabet_size, b, t, u, blank_);
            Tp emit = betas[(t+1) * U + u+1] + logp(denom, acts, start_idx, U, alphabet_size, b, t, u, labels[u]);
            betas[t * U + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
        }
        __syncthreads();
    }

    if (u == 0) { // u == 0 specifically is not important, but this only has to be done once.
        llBackward[b] = betas[0];
    }
}

template<typename Tp>
__global__ void compute_betas_kernel_naive(const Tp* const acts, const Tp* const denom, Tp* betas, Tp* llBackward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int* const start_indices, const int maxU, const int alphabet_size, const int blank_) {
    int tid = threadIdx.x; // mb
    const int T = xlen[tid];
    const int U = ylen[tid] + 1;
    const int* labels = mlabels + tid * (maxU - 1);
    const int start_idx = start_indices[tid];
    betas += start_idx;
    betas[T*U - 1] = logp(denom, acts, start_idx, U, alphabet_size, tid, T-1, U-1, blank_);
    for (int u = 0; u < U-1; ++u) {
        betas[(T-1)*U + u] = rnnt_helper::neg_inf<Tp>();
    }

    for (int t = T-2; t >=0; --t) {
        betas[t * U + U-1] = betas[(t+1) * U + U-1] + logp(denom, acts, start_idx, U, alphabet_size, tid, t, U-1, blank_);  // u = U-1
        for (int u = U-2; u >= 0; --u) {
            Tp no_emit = betas[(t+1) * U + u] + logp(denom, acts, start_idx, U, alphabet_size, tid, t, u, blank_);
            Tp emit = betas[(t+1) * U + u+1] + logp(denom, acts, start_idx, U, alphabet_size, tid, t, u, labels[u]);
            betas[t * U + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
        }
    }

    llBackward[tid] = betas[0];
}

template<int NT, typename Tp>
__global__ void compute_grad_kernel(Tp* grads, const Tp* const acts, const Tp* const denom, const Tp* alphas, const Tp* betas, const Tp* const logll, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int* const start_indices, const int maxU, const int alphabet_size, const int blank_) {
    int tid = threadIdx.x; // alphabet dim
    int v = tid;
    int col = blockIdx.x; // mb, t, u

    int mb = 0;
    while (mb < minibatch - 1 && start_indices[mb + 1] <= col) ++mb;
    
    const int T = xlen[mb];
    const int U = ylen[mb] + 1;
    const int* labels = mlabels + mb * (maxU - 1);

    int tu = col - start_indices[mb];
    int u = tu % U;
    int t = (tu - u) / U;

    while (v < alphabet_size) {
        Tp grad = 0;

        if (t < T-1) {
            Tp logpk = denom[col] + acts[col * alphabet_size + v];

            grad = exp(logpk + alphas[col] + betas[col] - logll[mb]);  // alphas[col] = alpha(t, u); betas[col] = beta(t, u)

            if (v == blank_) {
                grad -= exp(logpk + alphas[col] + betas[col + U] - logll[mb]);  // betas[col + U] = beta(t+1, u)
            } else if (v == labels[u] && u < U-1) {
                grad -= exp(logpk + alphas[col] + betas[col + U + 1] - logll[mb]); // betas[col + U + 1] = beta(t+1, u+1)
            }
        } else if (u == U-1 && v != blank_) {
            grad = exp(denom[col] + acts[col * alphabet_size + v]);
        }
        grads[col * alphabet_size + v] = grad;

        v += NT;
    }
}
