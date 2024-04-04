#ifndef MONOTONIC_RNNT_GPU_WORKSPACE_MANAGER_H
#define MONOTONIC_RNNT_GPU_WORKSPACE_MANAGER_H

#include <vector>
#include <algorithm>
#include <cassert>
#include "options.h"
#include "reduce.h"

template<typename dtype>
class GPURNNTWorkspaceManager {
public:

    /** For a given set of minibatch sample shapes manager the required
     *  workspace. Can calculate required size for all variables and
     *  performs structuring and access handling inside the allocated space
     *  after it is passed. Also handles memory access for activations and labels.
     *
     * \param [in]  acts 1-D flattened array containing all the model logits in packed row-major order
     * \param [in]  labels 1-D flattened array containing all the labels in non-packed row-major order
     * \param [in]  B Number of examples in minibatch.
     * \param [in]  T Number of time steps for each minibatch sample
     * \param [in]  S Number of labels for each minibatch sample
     * \param [in]  V alphabet size including blank
     *
     * \return Status information
     **/
    explicit GPURNNTWorkspaceManager(const dtype *const acts, const int *const labels, const int *B, const int *T,
                                     const int *S, const int *V)
            : B(B), T(T), S(S), S_max(nullptr), V(V), workspace_(nullptr), acts(acts), labels(labels),
              denom_start_indices(nullptr), var_start_offsets(nullptr), alphas(B), betas(B),
              denom(nullptr), dtype_size_(sizeof(dtype)) {
    }

    const int *T;  // device
    const int *S;  // device
    const int *B;  // device
    const int *V;  // device

    const dtype *const acts;  // device
    const int *const labels;  // device

    dtype *denom;  // workspace
    dtype *alphas;  // workspace
    dtype *betas;  // workspace

    int *denom_start_indices;  // workspace
    int *var_start_offsets;  // workspace

    const int *S_max;  // workspace

    dtype *ll_forward;  // workspace
    dtype *ll_backward;  // workspace

    [[nodiscard]] std::vector<int> T_host(CUstream stream) {
        int B_h = B_host(stream);
        std::vector<int> T_h(B_h);
        cudaMemcpyFromSymbolAsync(T_h.data(), T, B_h * sizeof(int), 0, cudaMemcpyDeviceToHost, stream);
        return T_h;
    }

    [[nodiscard]] std::vector<int> S_host(CUstream stream) const {
        int B_h = B_host(stream);
        std::vector<int> S_h(B_h);
        cudaMemcpyFromSymbolAsync(S_h.data(), S, B_host() * sizeof(int), 0, cudaMemcpyDeviceToHost, stream);
        return S_h;
    }

    [[nodiscard]] int B_host(CUstream stream) const {
        int B_h;
        cudaMemcpyFromSymbolAsync(&B_h, B, sizeof(int), 0, cudaMemcpyDeviceToHost, stream);
        return B_h;
    }

    [[nodiscard]] int V_host(CUstream stream) const {
        int V_h;
        cudaMemcpyFromSymbolAsync(&V_h, V, sizeof(int), 0, cudaMemcpyDeviceToHost, stream);
        return V_h;
    }

    [[nodiscard]] int num_denoms() const {
        int B_h = B_host();
        auto T_h = T_host();
        auto S_h = S_host();

        int result = 0;
        for (int b = 0; b < B_h; ++b) {
            result += T_h[b] * S_h[b];
        }
        return result;
    }

    [[nodiscard]] int fwd_bwd_var_positions(CUstream stream) const {
        int B_h = B_host(stream);
        auto T_h = T_host(stream);
        auto S_h = S_host(stream);

        int fwd_bwd_var_positions = 0;
        for (int b = 0; b < B_h; ++b) {
            fwd_bwd_var_positions += (T_h[b] + 1 - S_h[b]) * (S_h[b] + 1) - 1;
        }
    }

    [[nodiscard]] std::vector<dtype> acts_host(CUstream stream) {
        std::vector<dtype> acts_h(num_denoms(stream) * V_host(stream));
        cudaMemcpyFromSymbolAsync(acts_h.data(), acts, dtype_size_ * acts_h.size(), 0, cudaMemcpyDeviceToHost, stream);
        return acts_h;
    }

    [[nodiscard]] std::vector<dtype> denom_host(CUstream stream) {
        std::vector<dtype> denom_h(num_denoms(stream));
        cudaMemcpyFromSymbolAsync(denom_h.data(), denom, dtype_size_ * denom_h.size(), 0, cudaMemcpyDeviceToHost,
                                  stream);
        return denom_h;
    }

    [[nodiscard]] std::vector<dtype> alphas_host(CUstream stream) {
        std::vector<dtype> alphas_h(fwd_bwd_var_positions(stream));
        cudaMemcpyFromSymbolAsync(alphas_h.data(), alphas, dtype_size_ * alphas_h.size(), 0, cudaMemcpyDeviceToHost,
                                  stream);
        return alphas_h;
    }

    [[nodiscard]] std::vector<dtype> betas_host(CUstream stream) {
        std::vector<dtype> betas_h(fwd_bwd_var_positions(stream));
        cudaMemcpyFromSymbolAsync(betas_h.data(), betas, dtype_size_ * betas_h.size(), 0, cudaMemcpyDeviceToHost,
                                  stream);
        return betas_h;
    }

    [[nodiscard]] std::vector<dtype> ll_forward_host(CUstream stream) {
        int B_h = B_host(stream);
        std::vector<dtype> ll_forward_h(B_h);
        cudaMemcpyFromSymbolAsync(ll_forward_h.data(), ll_forward, dtype_size_ * B_h, 0, cudaMemcpyDeviceToHost,
                                  stream);
        return ll_forward_h;
    }

    [[nodiscard]] std::vector<dtype> ll_backward_host(CUstream stream) {
        int B_h = B_host(stream);
        std::vector<dtype> ll_backward_h(B_h);
        cudaMemcpyFromSymbolAsync(ll_backward_h.data(), ll_backward, dtype_size_ * B_h, 0, cudaMemcpyDeviceToHost,
                                  stream);
        return ll_backward_h;
    }

    /**
     * Calculate required memory for denominator, alphas and betas.
     * This memory needs to be allocated externally.
     *
     * \param [out] size_bytes Pointer to a scalar where the memory
     *              requirement in bytes will be placed.
     **/
    RNNTStatus get_workspace_size(size_t *size_bytes) const {
        int B_h = B_host();
        auto T_h = T_host();
        auto S_h = S_host();

        if (B_h <= 0) {
            return RNNT_STATUS_INVALID_VALUE;
        }
        for (int b = 0; b < B_h; ++b) {
            if (T_h[b] <= 0 || S_h[b] < 0 || T_h[b] < S_h[b]) {
                return RNNT_STATUS_INVALID_VALUE;
            }
        }

        *size_bytes = dtype_size_ * num_denoms()  // denom
                      + 2 * dtype_size_ * fwd_bwd_var_positions()  // alpha+beta
                      + 2 * B_h * sizeof(int)  // var_start_offsets + denom_start_indices
                      + sizeof(int)  // S_max
                      + 2 * B_h * dtype_size_;  // ll_forward + ll_backward

        return RNNT_STATUS_SUCCESS;
    }

    void set_workspace(void *workspace, CUstream stream) {
        workspace_ = workspace;

        int B_h = B_host(stream);
        auto T_h = T_host(stream);
        auto S_h = S_host(stream);
        int V_h = V_host(stream);

        int var_start_offsets_host[B_h + 1];
        var_start_offsets_host[0] = 0;
        for (int b = 1; b <= B_h; ++b) {
            var_start_offsets_host[b] = var_start_offsets_host[b - 1] + ((T_h[b] + 1 - S_h[b]) * (S_h[b] + 1) - 1);
        }

        int denom_start_indices_host[B_h + 1];
        denom_start_indices_host[0] = 0;
        for (int b = 1; b <= B_h; ++b) {
            denom_start_indices_host[b] = denom_start_indices_host[b - 1] + T_h[b - 1] * (S_h[b - 1] + 1) * V_h;
        }

        size_t current_offset = 0ul;

        denom = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += dtype_size_ * denom_start_indices_host[B_h] / V_h;

        alphas = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += dtype_size_ * var_start_offsets_host[B_h];

        betas = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += dtype_size_ * var_start_offsets_host[B_h];

        denom_start_indices = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += B_h * sizeof(int);
        cudaMemcpyToSymbolAsync(denom_start_indices, denom_start_indices_host, B_h * sizeof(int), 0,
                                cudaMemcpyHostToDevice,
                                stream);

        var_start_offsets = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += B_h * sizeof(int);
        cudaMemcpyToSymbolAsync(var_start_offsets, var_start_offsets_host, B_h * sizeof(int), 0, cudaMemcpyHostToDevice,
                                stream);

        int S_max_host = *std::max_element(S_h, S_h + B_h);
        S_max = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += sizeof(int);
        cudaMemcpyToSymbolAsync(S_max, &S_max_host, sizeof(int), 0, cudaMemcpyHostToDevice, stream);

        ll_forward = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += dtype_size_ * B_h;

        ll_backward = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += dtype_size_ * B_h;
    }

private:
    size_t dtype_size_;  // host

    void *workspace_;  // workspace
};


inline __device__ int alpha_idx(const int t, const int s, const int T, const int S) {
    // see cpu workspace manager for more detailed explanation. This is broken down for efficiency.
    int left_portion = t < S - 1 ? t : S - 1;
    int mid_portion = t < S - 1 ? 0 : (t < T - S ? t + 1 - S : T + 1 - 2 * S);
    int right_portion = t < T - S ? 0 : t - T + S;

    int left_sum = (left_portion + 1) * (left_portion + 2) / 2 - 1;
    int mid_sum = mid_portion * (S + 1);
    int right_sum = right_portion * (S + 1 - (right_portion + 1) / 2);

    int height_offset = t < T - S ? 0 : -right_portion - 1;

    return left_sum + mid_sum + right_sum + height_offset + s;
}

inline HOSTDEVICE int beta_idx(const int t, const int s, const int T, const int S) {
    return alpha_idx(T - t, S - s, T, S);
}

inline HOSTDEVICE int alpha_s_min(const int t, const int T, const int S) {
    return t < T - 1 - S ? 0 : t - (T - 1 - S);
}

inline HOSTDEVICE int alpha_s_max(const int t, const int T, const int S) {
    return t < S ? t + 1 : S;
}

inline HOSTDEVICE int beta_s_min(const int t, const int T, const int S) {
    return t < T - S ? 0 : t - (T - S);
}

inline HOSTDEVICE int beta_s_max(const int t, const int T, const int S) {
    return t < S ? t : S;
}

template<typename dtype>
inline HOSTDEVICE dtype alpha(dtype *alphas, const int t, const int s, const int T, const int S) {
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

template<typename dtype>
inline HOSTDEVICE dtype beta(dtype *betas, const int t, const int s, const int T, const int S) {
    // Note: t = T and s = S+1 are allowed to be used as virtual starts
    // They correspond to constants
    if (s == S + 1) {
        return rnnt_helper::neg_inf<dtype>();
    }

    if (t == T) {
        return s == 0 ? 0 : rnnt_helper::neg_inf<dtype>();
    }

    if (s > t || S - s - 1 > T - 1 - t) {
        return rnnt_helper::neg_inf<dtype>();
    }

    return betas[beta_idx(t, s, T, S)];
}

inline HOSTDEVICE int denom_idx(const int t, const int s, const int S) {
    return t * (S + 1) + s;
}

inline HOSTDEVICE int act_idx(const int t, const int s, const int v, const int S, const int V) {
    return (denom_idx(t, s, S)) * V + v;
}

template<typename dtype>
inline HOSTDEVICE dtype
log_p(dtype *acts, dtype *denom, const int t, const int s, const int v, const int S, const int V) {
    return acts[act_idx(t, s, v, S, V)] + denom[denom_idx(t, s, S)];
}

#endif //MONOTONIC_RNNT_GPU_WORKSPACE_MANAGER_H
