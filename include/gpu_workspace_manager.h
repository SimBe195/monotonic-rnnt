#ifndef MONOTONIC_RNNT_GPU_WORKSPACE_MANAGER_H
#define MONOTONIC_RNNT_GPU_WORKSPACE_MANAGER_H

#include <vector>
#include <algorithm>
#include <cassert>
#include "status.h"
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
    explicit GPURNNTWorkspaceManager(const dtype *const acts, const int *const labels, const int B, const int *T,
                                     const int *S, const int V)
            : B_h(B), B(nullptr), T(T), S(S), S_max(nullptr), V_h(V), V(nullptr), acts(acts), labels(labels),
              denom_start_indices(nullptr), var_start_offsets(nullptr), alphas(nullptr), betas(nullptr),
              denom(nullptr), dtype_size_(sizeof(dtype)) {
    }

    const int B_h; // host
    const int V_h; // host

    const int *T;  // device
    const int *S;  // device
    int *B;  // device
    int *V;  // device

    const dtype *const acts;  // device
    const int *const labels;  // device

    dtype *denom;  // workspace
    dtype *alphas;  // workspace
    dtype *betas;  // workspace

    int *denom_start_indices;  // workspace
    int *var_start_offsets;  // workspace

    int *S_max;  // workspace

    dtype *ll_forward;  // workspace
    dtype *ll_backward;  // workspace

    [[nodiscard]] std::vector<int> T_host(CUstream stream, bool sync = true) const {
        std::vector<int> T_h(B_h);
        cudaMemcpyAsync(T_h.data(), T, B_h * sizeof(int), cudaMemcpyDeviceToHost, stream);
        if (sync) {
            cudaStreamSynchronize(stream);
        }
        return T_h;
    }

    [[nodiscard]] std::vector<int> S_host(CUstream stream, bool sync = true) const {
        std::vector<int> S_h(B_h);
        cudaMemcpyAsync(S_h.data(), S, B_h * sizeof(int), cudaMemcpyDeviceToHost, stream);
        if (sync) {
            cudaStreamSynchronize(stream);
        }
        return S_h;
    }

    [[nodiscard]] int B_host() const {
        return B_h;
    }

    [[nodiscard]] int V_host() const {
        return V_h;
    }

    [[nodiscard]] int num_denoms(CUstream stream) const {
        auto T_h = T_host(stream, false);
        auto S_h = S_host(stream, false);
        cudaStreamSynchronize(stream);

        int result = 0;
        for (int b = 0; b < B_h; ++b) {
            result += T_h[b] * (S_h[b] + 1);
        }
        return result;
    }

    [[nodiscard]] int num_fwd_bwd_var_positions(CUstream stream) const {
        auto T_h = T_host(stream, false);
        auto S_h = S_host(stream, false);
        cudaStreamSynchronize(stream);

        int fwd_bwd_var_positions = 0;
        for (int b = 0; b < B_h; ++b) {
            fwd_bwd_var_positions += (T_h[b] + 1 - S_h[b]) * (S_h[b] + 1) - 1;
        }

        return fwd_bwd_var_positions;
    }

    [[nodiscard]] std::vector<int> var_start_offsets_host(CUstream stream, bool sync = true) {
        std::vector<int> var_start_offsets_h(B_h);
        cudaMemcpyAsync(var_start_offsets_h.data(), var_start_offsets, sizeof(int) * var_start_offsets_h.size(),
                        cudaMemcpyDeviceToHost, stream);
        if (sync) {
            cudaStreamSynchronize(stream);
        }
        return var_start_offsets_h;

    }

    [[nodiscard]] std::vector<dtype> acts_host(CUstream stream, bool sync = true) {
        std::vector<dtype> acts_h(num_denoms(stream) * V_host());
        cudaMemcpyAsync(acts_h.data(), acts, dtype_size_ * acts_h.size(), cudaMemcpyDeviceToHost, stream);
        if (sync) {
            cudaStreamSynchronize(stream);
        }
        return acts_h;
    }

    [[nodiscard]] std::vector<dtype> denom_host(CUstream stream, bool sync = true) {
        std::vector<dtype> denom_h(num_denoms(stream));
        cudaMemcpyAsync(denom_h.data(), denom, dtype_size_ * denom_h.size(), cudaMemcpyDeviceToHost,
                        stream);
        if (sync) {
            cudaStreamSynchronize(stream);
        }
        return denom_h;
    }

    [[nodiscard]] std::vector<dtype> alphas_host(CUstream stream, bool sync = true) {
        std::vector<dtype> alphas_h(num_fwd_bwd_var_positions(stream));
        cudaMemcpyAsync(alphas_h.data(), alphas, dtype_size_ * alphas_h.size(), cudaMemcpyDeviceToHost,
                        stream);
        if (sync) {
            cudaStreamSynchronize(stream);
        }
        return alphas_h;
    }

    [[nodiscard]] std::vector<dtype> betas_host(CUstream stream, bool sync = true) {
        std::vector<dtype> betas_h(num_fwd_bwd_var_positions(stream));
        cudaMemcpyAsync(betas_h.data(), betas, dtype_size_ * betas_h.size(), cudaMemcpyDeviceToHost,
                        stream);
        if (sync) {
            cudaStreamSynchronize(stream);
        }
        return betas_h;
    }

    [[nodiscard]] std::vector<dtype> ll_forward_host(CUstream stream, bool sync = true) {
        std::vector<dtype> ll_forward_h(B_h);
        cudaMemcpyAsync(ll_forward_h.data(), ll_forward, dtype_size_ * B_h, cudaMemcpyDeviceToHost,
                        stream);
        if (sync) {
            cudaStreamSynchronize(stream);
        }
        return ll_forward_h;
    }

    [[nodiscard]] std::vector<dtype> ll_backward_host(CUstream stream, bool sync = true) {
        std::vector<dtype> ll_backward_h(B_h);
        cudaMemcpyAsync(ll_backward_h.data(), ll_backward, dtype_size_ * B_h, cudaMemcpyDeviceToHost,
                        stream);
        if (sync) {
            cudaStreamSynchronize(stream);
        }
        return ll_backward_h;
    }

    /**
     * Calculate required memory for denominator, alphas and betas.
     * This memory needs to be allocated externally.
     *
     * \param [out] size_bytes Pointer to a scalar where the memory
     *              requirement in bytes will be placed.
     **/
    RNNTStatus get_workspace_size(size_t *size_bytes, CUstream stream) const {
        auto T_h = T_host(stream);
        auto S_h = S_host(stream);

        if (B_h <= 0) {
            return RNNT_STATUS_INVALID_VALUE;
        }
        for (int b = 0; b < B_h; ++b) {
            if (T_h[b] <= 0 || S_h[b] < 0 || T_h[b] < S_h[b]) {
                return RNNT_STATUS_INVALID_VALUE;
            }
        }

        *size_bytes = dtype_size_ * num_denoms(stream)  // denom
                      + 2 * dtype_size_ * num_fwd_bwd_var_positions(stream)  // alpha+beta
                      + 2 * B_h * sizeof(int)  // var_start_offsets + denom_start_indices
                      + 2 * B_h * dtype_size_  // ll_forward + ll_backward
                      + 3 * sizeof(int);  // B, V, S_max

        return RNNT_STATUS_SUCCESS;
    }

    void set_workspace(void *workspace, CUstream stream) {

        auto T_h = T_host(stream, false);
        auto S_h = S_host(stream, false);
        cudaStreamSynchronize(stream);

        int var_start_offsets_host[B_h + 1];
        var_start_offsets_host[0] = 0;
        for (int b = 1; b <= B_h; ++b) {
            var_start_offsets_host[b] =
                    var_start_offsets_host[b - 1] + ((T_h[b - 1] + 1 - S_h[b - 1]) * (S_h[b - 1] + 1) - 1);
        }

        int denom_start_indices_host[B_h + 1];
        denom_start_indices_host[0] = 0;
        for (int b = 1; b <= B_h; ++b) {
            denom_start_indices_host[b] = denom_start_indices_host[b - 1] + T_h[b - 1] * (S_h[b - 1] + 1);
        }

        size_t current_offset = 0ul;

        denom = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += dtype_size_ * denom_start_indices_host[B_h];

        alphas = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += dtype_size_ * var_start_offsets_host[B_h];

        betas = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += dtype_size_ * var_start_offsets_host[B_h];

        denom_start_indices = reinterpret_cast<int *>(static_cast<char *>(workspace) + current_offset);
        current_offset += B_h * sizeof(int);
        cudaMemcpyAsync(denom_start_indices, denom_start_indices_host, B_h * sizeof(int),
                        cudaMemcpyHostToDevice,
                        stream);

        var_start_offsets = reinterpret_cast<int *>(static_cast<char *>(workspace) + current_offset);
        current_offset += B_h * sizeof(int);
        cudaMemcpyAsync(var_start_offsets, var_start_offsets_host, B_h * sizeof(int), cudaMemcpyHostToDevice,
                        stream);

        ll_forward = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += dtype_size_ * B_h;

        ll_backward = reinterpret_cast<dtype *>(static_cast<char *>(workspace) + current_offset);
        current_offset += dtype_size_ * B_h;

        B = reinterpret_cast<int *>(static_cast<char *>(workspace) + current_offset);
        current_offset += sizeof(int);
        cudaMemcpyAsync(B, &B_h, sizeof(int), cudaMemcpyHostToDevice, stream);

        V = reinterpret_cast<int *>(static_cast<char *>(workspace) + current_offset);
        current_offset += sizeof(int);
        cudaMemcpyAsync(V, &V_h, sizeof(int), cudaMemcpyHostToDevice, stream);

        int S_max_h = *std::max_element(S_h.begin(), S_h.end());
        S_max = reinterpret_cast<int *>(static_cast<char *>(workspace) + current_offset);
        cudaMemcpyAsync(S_max, &S_max_h, sizeof(int), cudaMemcpyHostToDevice, stream);

        cudaStreamSynchronize(stream);
    }

private:
    size_t dtype_size_;  // host
};


inline HOSTDEVICE int alpha_idx(const int t, const int s, const int T, const int S) {
    // see cpu workspace manager for more detailed explanation. This is broken down for efficiency.
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

inline HOSTDEVICE int alpha_s_min(const int t, const int T, const int S) {
    return t < T - 1 - S ? 0 : t - (T - 1 - S);
}

inline HOSTDEVICE int alpha_s_max(const int t, const int S) {
    return t < S ? t + 1 : S;
}

inline HOSTDEVICE int beta_s_min(const int t, const int T, const int S) {
    return t < T - S ? 0 : t - (T - S);
}

inline HOSTDEVICE int beta_s_max(const int t, const int S) {
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
        return s == S ? 0 : rnnt_helper::neg_inf<dtype>();
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
log_p(const dtype *const acts, const dtype *const denom, const int t, const int s, const int v, const int S,
      const int V) {
    return acts[act_idx(t, s, v, S, V)] + denom[denom_idx(t, s, S)];
}

#endif //MONOTONIC_RNNT_GPU_WORKSPACE_MANAGER_H
