#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "gpu_rnnt.h"
#include "gpu_workspace_manager.h"
#include "test.h"

template <typename T>
void vector_to_gpu(T *&gpu_space, std::vector<T> &vec) {
    cudaMalloc(&gpu_space, vec.size() * sizeof(T));
    cudaMemcpy(gpu_space, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice);
}

bool fwd_test() {
    // Similar to example in README file
    int B = 1;
    int T = 4;
    int S = 2;
    int V = 3;

    std::vector<int> labels = {1, 2};

    std::vector<int> lengths = {T};
    std::vector<int> label_lengths = {S};

    std::vector<float> probs = {
        // t = 0
        0.6, 0.3, 0.1,  // s = 0
        0.7, 0.1, 0.2,  // s = 1
        0.5, 0.1, 0.4,  // s = 2

        // t = 1
        0.5, 0.4, 0.1,  // s = 0
        0.5, 0.1, 0.4,  // s = 1
        0.8, 0.1, 0.1,  // s = 2

        // t = 2
        0.4, 0.3, 0.3,  // s = 0
        0.5, 0.1, 0.4,  // s = 1
        0.7, 0.2, 0.1,  // s = 2

        // t = 3
        0.8, 0.1, 0.1,  // s = 0
        0.3, 0.1, 0.6,  // s = 1
        0.8, 0.1, 0.1   // s = 2
    };

    std::vector<float> logits(probs.size());
    std::transform(probs.begin(), probs.end(), logits.begin(), [](float v) { return std::log(v); });

    float *logits_gpu;
    vector_to_gpu(logits_gpu, logits);
    int *labels_gpu;
    vector_to_gpu(labels_gpu, labels);
    int *lengths_gpu;
    vector_to_gpu(lengths_gpu, lengths);
    int *label_lengths_gpu;
    vector_to_gpu(label_lengths_gpu, label_lengths);

    GpuRNNTWorkspaceManager<float> workspace_manager(logits_gpu, labels_gpu, B, lengths_gpu, label_lengths_gpu, V);
    throw_on_error(workspace_manager.create_workspace(), "Error: get_workspace_size in fwd_test");

    CUstream stream;
    cudaStreamCreate(&stream);

    float score_fwd;
    GpuRNNTComputer<float> rnnt_computer(workspace_manager, 0, stream);

    throw_on_error(rnnt_computer.cost(&score_fwd), "Error: compute_rnnt_loss forward in fwd_test");

    workspace_manager.free_workspace();

    cudaFree(logits_gpu);
    cudaFree(labels_gpu);
    cudaFree(lengths_gpu);
    cudaFree(label_lengths_gpu);

    cudaStreamDestroy(stream);

    return rnnt_helper::is_close(score_fwd, static_cast<float>(-log(0.363)));
}

bool bwd_test() {
    // Similar to example in README file
    int B = 1;
    int T = 4;
    int S = 2;
    int V = 3;

    std::vector<int> labels = {1, 2};

    std::vector<int> lengths = {T};
    std::vector<int> label_lengths = {S};

    std::vector<float> probs = {
        // t = 0
        0.6, 0.3, 0.1,  // s = 0
        0.7, 0.1, 0.2,  // s = 1
        0.5, 0.1, 0.4,  // s = 2

        // t = 1
        0.5, 0.4, 0.1,  // s = 0
        0.5, 0.1, 0.4,  // s = 1
        0.8, 0.1, 0.1,  // s = 2

        // t = 2
        0.4, 0.3, 0.3,  // s = 0
        0.5, 0.1, 0.4,  // s = 1
        0.7, 0.2, 0.1,  // s = 2

        // t = 3
        0.8, 0.1, 0.1,  // s = 0
        0.3, 0.1, 0.6,  // s = 1
        0.8, 0.1, 0.1   // s = 2
    };

    std::vector<float> logits(probs.size());
    std::transform(probs.begin(), probs.end(), logits.begin(), [](float v) { return std::log(v); });

    float *logits_gpu;
    vector_to_gpu(logits_gpu, logits);
    int *labels_gpu;
    vector_to_gpu(labels_gpu, labels);
    int *lengths_gpu;
    vector_to_gpu(lengths_gpu, lengths);
    int *label_lengths_gpu;
    vector_to_gpu(label_lengths_gpu, label_lengths);

    GpuRNNTWorkspaceManager<float> workspace_manager(logits_gpu, labels_gpu, B, lengths_gpu, label_lengths_gpu, V);
    throw_on_error(workspace_manager.create_workspace(), "Error: get_workspace_size in bwd_test");

    CUstream stream;
    cudaStreamCreate(&stream);

    float score_fwd;
    GpuRNNTComputer<float> rnnt_computer(workspace_manager, 0, stream);
    throw_on_error(rnnt_computer.cost(&score_fwd), "Error: compute_rnnt_loss forward in bwd_test");

    float *grads;
    cudaMalloc(&grads, sizeof(float) * logits.size());

    float score_bwd;
    throw_on_error(rnnt_computer.cost_and_grad(&score_bwd, grads),
                   "Error: compute_rnnt_loss forward+backward in bwd_test");

    workspace_manager.free_workspace();
    cudaFree(grads);
    cudaFree(logits_gpu);
    cudaFree(labels_gpu);
    cudaFree(lengths_gpu);
    cudaFree(label_lengths_gpu);

    cudaStreamDestroy(stream);

    return rnnt_helper::is_close(score_fwd, score_bwd);
}

bool grads_test() {
    // Similar to example in README file
    int B = 1;
    int T = 4;
    int S = 2;
    int V = 3;

    std::vector<int> labels = {1, 2};

    std::vector<int> lengths = {T};
    std::vector<int> label_lengths = {S};

    std::vector<float> probs = {
        // t = 0
        0.6, 0.3, 0.1,  // s = 0
        0.7, 0.1, 0.2,  // s = 1
        0.5, 0.1, 0.4,  // s = 2

        // t = 1
        0.5, 0.4, 0.1,  // s = 0
        0.5, 0.1, 0.4,  // s = 1
        0.8, 0.1, 0.1,  // s = 2

        // t = 2
        0.4, 0.3, 0.3,  // s = 0
        0.5, 0.1, 0.4,  // s = 1
        0.7, 0.2, 0.1,  // s = 2

        // t = 3
        0.8, 0.1, 0.1,  // s = 0
        0.3, 0.1, 0.6,  // s = 1
        0.8, 0.1, 0.1   // s = 2
    };

    std::vector<float> logits(probs.size());
    std::transform(probs.begin(), probs.end(), logits.begin(), [](float v) { return std::log(v); });

    float *logits_gpu;
    vector_to_gpu(logits_gpu, logits);
    int *labels_gpu;
    vector_to_gpu(labels_gpu, labels);
    int *lengths_gpu;
    vector_to_gpu(lengths_gpu, lengths);
    int *label_lengths_gpu;
    vector_to_gpu(label_lengths_gpu, label_lengths);

    GpuRNNTWorkspaceManager<float> workspace_manager(logits_gpu, labels_gpu, B, lengths_gpu, label_lengths_gpu, V);
    throw_on_error(workspace_manager.create_workspace(), "Error: get_workspace_size in grads_test");

    CUstream stream;
    cudaStreamCreate(&stream);

    float *grads;
    cudaMalloc(&grads, sizeof(float) * logits.size());
    float score;
    GpuRNNTComputer<float> rnnt_computer(workspace_manager, 0, stream);
    throw_on_error(rnnt_computer.cost_and_grad(&score, grads),
                   "Error: compute_rnnt_loss forward+backward in grads_test");

    std::vector<float> expected_grads = {
        // t = 0
        0.04, -0.14, 0.1,  // s = 0
        0.0, 0.0, 0.0,     // s = 1
        0.0, 0.0, 0.0,     // s = 2

        // t = 1
        0.13, -0.19, 0.06,   // s = 0
        -0.04, 0.04, -0.01,  // s = 1
        0.0, 0.0, 0.0,       // s = 2

        // t = 2
        0.06, -0.1, 0.04,   // s = 0
        0.01, 0.07, -0.08,  // s = 1
        -0.06, 0.04, 0.02,  // s = 2

        // t = 3
        0.0, 0.0, 0.0,      // s = 0
        0.14, 0.05, -0.19,  // s = 1
        -0.11, 0.05, 0.05   // s = 2
    };

    std::vector<float> grads_host(expected_grads.size());
    cudaMemcpy(grads_host.data(), grads, sizeof(float) * grads_host.size(), cudaMemcpyDeviceToHost);

    bool grads_close = true;

    for (size_t idx = 0ul; idx < expected_grads.size(); ++idx) {
        grads_close &= std::abs(expected_grads[idx] - grads_host[idx]) < 1e-02;
    }

    workspace_manager.free_workspace();
    cudaFree(grads);
    cudaFree(logits_gpu);
    cudaFree(labels_gpu);
    cudaFree(lengths_gpu);
    cudaFree(label_lengths_gpu);

    cudaStreamDestroy(stream);

    return grads_close;
}

bool multibatch_test() {
    int B = 2;
    int V = 3;

    std::vector<int> labels = {1, 0, 1, 2};

    std::vector<int> lengths = {2, 4};
    std::vector<int> label_lengths = {1, 2};

    std::vector<float> probs = {
        // b = 0
        // t = 0
        0.6, 0.3, 0.1,  // s = 0
        0.7, 0.1, 0.2,  // s = 1

        // t = 1
        0.5, 0.4, 0.1,  // s = 0
        0.5, 0.1, 0.4,  // s = 1

        // b = 1
        // t = 0
        0.6, 0.3, 0.1,  // s = 0
        0.7, 0.1, 0.2,  // s = 1
        0.5, 0.1, 0.4,  // s = 2

        // t = 1
        0.5, 0.4, 0.1,  // s = 0
        0.5, 0.1, 0.4,  // s = 1
        0.8, 0.1, 0.1,  // s = 2

        // t = 2
        0.4, 0.3, 0.3,  // s = 0
        0.5, 0.1, 0.4,  // s = 1
        0.7, 0.2, 0.1,  // s = 2

        // t = 3
        0.8, 0.1, 0.1,  // s = 0
        0.3, 0.1, 0.6,  // s = 1
        0.8, 0.1, 0.1   // s = 2
    };

    std::vector<float> logits(probs.size());
    std::transform(probs.begin(), probs.end(), logits.begin(), [](float v) { return std::log(v); });

    float *logits_gpu;
    vector_to_gpu(logits_gpu, logits);
    int *labels_gpu;
    vector_to_gpu(labels_gpu, labels);
    int *lengths_gpu;
    vector_to_gpu(lengths_gpu, lengths);
    int *label_lengths_gpu;
    vector_to_gpu(label_lengths_gpu, label_lengths);

    GpuRNNTWorkspaceManager<float> workspace_manager(logits_gpu, labels_gpu, B, lengths_gpu, label_lengths_gpu, V);
    throw_on_error(workspace_manager.create_workspace(), "Error: get_workspace_size in multibatch_test");

    CUstream stream;
    cudaStreamCreate(&stream);

    GpuRNNTComputer<float> rnnt_computer(workspace_manager, 0, stream);
    std::vector<float> scores_fwd(B);
    throw_on_error(rnnt_computer.cost(scores_fwd.data()), "Error: compute_rnnt_loss forward in multibatch_test");

    float *grads;
    cudaMalloc(&grads, sizeof(float) * logits.size());
    std::vector<float> scores_bwd(B);
    throw_on_error(rnnt_computer.cost_and_grad(scores_bwd.data(), grads),
                   "Error: compute_rnnt_loss forward+backward in multibatch_test");

    std::vector<float> expected_grads = {
        // b = 0
        // t = 0
        -0.02, -0.08, 0.1,  // s = 0
        0.0, 0.0, 0.0,      // s = 1

        // t = 1
        0.31, -0.37, 0.06,  // s = 0
        -0.19, 0.04, 0.15,  // s = 1

        // b = 1
        // t = 0
        0.04, -0.14, 0.1,  // s = 0
        0.0, 0.0, 0.0,     // s = 1
        0.0, 0.0, 0.0,     // s = 2

        // t = 1
        0.13, -0.19, 0.06,   // s = 0
        -0.04, 0.04, -0.01,  // s = 1
        0.0, 0.0, 0.0,       // s = 2

        // t = 2
        0.06, -0.1, 0.04,   // s = 0
        0.01, 0.07, -0.08,  // s = 1
        -0.06, 0.04, 0.02,  // s = 2

        // t = 3
        0.0, 0.0, 0.0,      // s = 0
        0.14, 0.05, -0.19,  // s = 1
        -0.11, 0.05, 0.05   // s = 2
    };

    std::vector<float> grads_host(expected_grads.size());
    cudaMemcpy(grads_host.data(), grads, sizeof(float) * grads_host.size(), cudaMemcpyDeviceToHost);

    bool grads_close = true;

    for (size_t idx = 0ul; idx < expected_grads.size(); ++idx) {
        grads_close &= std::abs(expected_grads[idx] - grads_host[idx]) < 1e-02;
    }

    workspace_manager.free_workspace();
    cudaFree(grads);
    cudaFree(logits_gpu);
    cudaFree(labels_gpu);
    cudaFree(lengths_gpu);
    cudaFree(label_lengths_gpu);

    cudaStreamDestroy(stream);

    return rnnt_helper::is_close(scores_fwd[0], static_cast<float>(-log(0.39))) &&
           rnnt_helper::is_close(scores_fwd[1], static_cast<float>(-log(0.363))) &&
           rnnt_helper::is_close(scores_fwd[0], scores_bwd[0]) && rnnt_helper::is_close(scores_fwd[1], scores_bwd[1]) &&
           grads_close;
}

bool infnan_test() {
    int B = 1;
    int T = 50;
    int S = 10;
    int V = 15;

    std::vector<int> labels = genLabels(V, S);
    std::vector<int> label_lengths = {S};

    std::vector<float> acts(B * T * (S + 1) * V);
    genActs(acts);

    std::vector<int> lengths = {T};

    float *acts_gpu;
    vector_to_gpu(acts_gpu, acts);
    int *labels_gpu;
    vector_to_gpu(labels_gpu, labels);
    int *lengths_gpu;
    vector_to_gpu(lengths_gpu, lengths);
    int *label_lengths_gpu;
    vector_to_gpu(label_lengths_gpu, label_lengths);

    GpuRNNTWorkspaceManager<float> workspace_manager(acts_gpu, labels_gpu, B, lengths_gpu, label_lengths_gpu, V);
    throw_on_error(workspace_manager.create_workspace(), "Error: get_workspace_size in infnan_test");

    CUstream stream;
    cudaStreamCreate(&stream);

    GpuRNNTComputer<float> rnnt_computer(workspace_manager, 0, stream);

    float cost;
    float *grads;
    cudaMalloc(&grads, sizeof(float) * acts.size());
    throw_on_error(rnnt_computer.cost_and_grad(&cost, grads), "Error: compute_rnnt_loss forward in infnan_test");

    bool status = true;
    status &= !std::isinf(cost);
    status &= !std::isnan(cost);

    std::vector<float> grads_host(acts.size());
    cudaMemcpy(grads_host.data(), grads, sizeof(float) * grads_host.size(), cudaMemcpyDeviceToHost);

    for (auto grad : grads_host) {
        status &= !std::isinf(grad);
        status &= !std::isnan(grad);
    }

    workspace_manager.free_workspace();
    cudaFree(grads);
    cudaFree(acts_gpu);
    cudaFree(labels_gpu);
    cudaFree(lengths_gpu);
    cudaFree(label_lengths_gpu);

    cudaStreamDestroy(stream);

    return status;
}

bool align_restrict_test() {
    const int B = 1;
    const int T = 4;
    const int S = 2;
    const int V = 3;

    std::vector<int> labels = {1, 2};

    std::vector<int> lengths = {T};
    std::vector<int> label_lengths = {S};

    std::vector<float> probs = {
        // t = 0
        0.6, 0.3, 0.1,  // s = 0
        0.7, 0.1, 0.2,  // s = 1
        0.5, 0.1, 0.4,  // s = 2

        // t = 1
        0.5, 0.4, 0.1,  // s = 0
        0.5, 0.1, 0.4,  // s = 1
        0.8, 0.1, 0.1,  // s = 2

        // t = 2
        0.4, 0.3, 0.3,  // s = 0
        0.5, 0.1, 0.4,  // s = 1
        0.7, 0.2, 0.1,  // s = 2

        // t = 3
        0.8, 0.1, 0.1,  // s = 0
        0.3, 0.1, 0.6,  // s = 1
        0.8, 0.1, 0.1   // s = 2
    };

    std::vector<int> alignment = {0, 1, 0, 2};

    std::vector<float> logits(probs.size());
    std::transform(probs.begin(), probs.end(), logits.begin(), [](float v) { return std::log(v); });

    float *logits_gpu;
    vector_to_gpu(logits_gpu, logits);
    int *labels_gpu;
    vector_to_gpu(labels_gpu, labels);
    int *lengths_gpu;
    vector_to_gpu(lengths_gpu, lengths);
    int *label_lengths_gpu;
    vector_to_gpu(label_lengths_gpu, label_lengths);
    int *alignment_gpu;
    vector_to_gpu(alignment_gpu, alignment);

    GpuRNNTWorkspaceManager<float> workspace_manager(logits_gpu, labels_gpu, B, lengths_gpu, label_lengths_gpu, V);

    throw_on_error(workspace_manager.create_workspace(), "Error: create_workspace in align_restrict_test");

    CUstream stream;
    cudaStreamCreate(&stream);

    float score_fwd;
    GpuRNNTComputer<float> rnnt_computer(workspace_manager, 0, stream);

    bool status = true;

    /*
     *  p(a|x) = p(0|_)p(0|_)p(1|_)p(2|1) +
     *           p(0|_)p(1|_)p(0|1)p(2|1) +
     *           p(0|_)p(1|_)p(2|1)p(0|2) +
     *           p(1|_)p(0|1)p(0|1)p(2|1) +
     *           p(1|_)p(0|1)p(2|1)p(0|2) +
     *           p(1|_)p(2|1)p(0|2)p(0|2)
     *         = 0.6 * 0.5 * 0.3 * 0.6 +
     *           0.6 * 0.4 * 0.5 * 0.6 +
     *           0.6 * 0.4 * 0.4 * 0.8 +
     *           0.3 * 0.5 * 0.5 * 0.6 +
     *           0.3 * 0.5 * 0.4 * 0.8 +
     *           0.3 * 0.4 * 0.7 * 0.8
     *         = 0.054 +
     *           0.072 +
     *           0.0768 +
     *           0.045 +
     *           0.048 +
     *           0.0672
     *         = 0.363
     */

    throw_on_error(rnnt_computer.cost(&score_fwd), "Error: rnnt_computer forward in align_restrict_test");
    if (!rnnt_helper::is_close(score_fwd, static_cast<float>(-log(0.363)))) {
        std::cout << "Unrestricted test failed." << std::endl;
        status = false;
    }

    // Allowing a shift of 2 doesn't remove any paths
    workspace_manager.restrict_to_alignment(alignment_gpu, 2, 0);
    throw_on_error(rnnt_computer.cost(&score_fwd), "Error: rnnt_computer forward in align_restrict_test");
    if (!rnnt_helper::is_close(score_fwd, static_cast<float>(-log(0.363)))) {
        std::cout << "Align-restricted test with 2 frame shift failed." << std::endl;
        status = false;
    }

    // Shift of 0 means sticking to alignment [0, 1, 0, 2]
    workspace_manager.restrict_to_alignment(alignment_gpu, 0, 0);
    throw_on_error(rnnt_computer.cost(&score_fwd), "Error: rnnt_computer forward in align_restrict_test");
    if (!rnnt_helper::is_close(score_fwd, static_cast<float>(-log(0.072)))) {
        std::cout << "Align-restricted test with 0 frame shift failed." << std::endl;
        status = false;
    }

    // Shift of 1 means removing only path [1, 2, 0, 0]
    workspace_manager.restrict_to_alignment(alignment_gpu, 1, 0);
    throw_on_error(rnnt_computer.cost(&score_fwd), "Error: rnnt_computer forward in align_restrict_test");
    if (!rnnt_helper::is_close(score_fwd, static_cast<float>(-log(0.2958)))) {
        std::cout << "Align-restricted test with 1 frame shift failed." << std::endl;
        status = false;
    }

    workspace_manager.free_workspace();

    cudaFree(logits_gpu);
    cudaFree(labels_gpu);
    cudaFree(lengths_gpu);
    cudaFree(label_lengths_gpu);

    cudaStreamDestroy(stream);

    return status;
}

bool align_restrict_multibatch_test() {
    const int B = 2;
    const int T = 4;
    const int S = 2;
    const int V = 3;

    std::vector<int> labels = {1, 2, 1, 2};

    std::vector<int> lengths = {T, T};
    std::vector<int> label_lengths = {S, S};

    std::vector<float> probs = {
        // b = 0
        // t = 0
        0.6, 0.3, 0.1,  // s = 0
        0.7, 0.1, 0.2,  // s = 1
        0.5, 0.1, 0.4,  // s = 2

        // t = 1
        0.5, 0.4, 0.1,  // s = 0
        0.5, 0.1, 0.4,  // s = 1
        0.8, 0.1, 0.1,  // s = 2

        // t = 2
        0.4, 0.3, 0.3,  // s = 0
        0.5, 0.1, 0.4,  // s = 1
        0.7, 0.2, 0.1,  // s = 2

        // t = 3
        0.8, 0.1, 0.1,  // s = 0
        0.3, 0.1, 0.6,  // s = 1
        0.8, 0.1, 0.1,  // s = 2

        // b = 1
        // t = 0
        0.6, 0.3, 0.1,  // s = 0
        0.7, 0.1, 0.2,  // s = 1
        0.5, 0.1, 0.4,  // s = 2

        // t = 1
        0.5, 0.4, 0.1,  // s = 0
        0.5, 0.1, 0.4,  // s = 1
        0.8, 0.1, 0.1,  // s = 2

        // t = 2
        0.4, 0.3, 0.3,  // s = 0
        0.5, 0.1, 0.4,  // s = 1
        0.7, 0.2, 0.1,  // s = 2

        // t = 3
        0.8, 0.1, 0.1,  // s = 0
        0.3, 0.1, 0.6,  // s = 1
        0.8, 0.1, 0.1   // s = 2
    };

    std::vector<int> alignment = {0, 1, 0, 2, 1, 2, 0, 0};

    std::vector<float> logits(probs.size());
    std::transform(probs.begin(), probs.end(), logits.begin(), [](float v) { return std::log(v); });

    float *logits_gpu;
    vector_to_gpu(logits_gpu, logits);
    int *labels_gpu;
    vector_to_gpu(labels_gpu, labels);
    int *lengths_gpu;
    vector_to_gpu(lengths_gpu, lengths);
    int *label_lengths_gpu;
    vector_to_gpu(label_lengths_gpu, label_lengths);
    int *alignment_gpu;
    vector_to_gpu(alignment_gpu, alignment);

    GpuRNNTWorkspaceManager<float> workspace_manager(logits_gpu, labels_gpu, B, lengths_gpu, label_lengths_gpu, V);

    throw_on_error(workspace_manager.create_workspace(), "Error: create_workspace in align_restrict_multibatch_test");

    CUstream stream;
    cudaStreamCreate(&stream);

    std::vector<float> scores_fwd(B);
    GpuRNNTComputer<float> rnnt_computer(workspace_manager, 0, stream);

    bool status = true;

    throw_on_error(rnnt_computer.cost(scores_fwd.data()), "Error: rnnt_computer forward in align_restrict_multibatch_test");
    if (!(rnnt_helper::is_close(scores_fwd[0], static_cast<float>(-log(0.363))) &&
          rnnt_helper::is_close(scores_fwd[1], static_cast<float>(-log(0.363))))) {
        std::cout << "Unrestricted multibatch test failed." << std::endl;
        status = false;
    }

    // Allowing a shift of 3 doesn't remove any paths
    workspace_manager.restrict_to_alignment(alignment_gpu, 2, 0);
    throw_on_error(rnnt_computer.cost(scores_fwd.data()), "Error: rnnt_computer forward in align_restrict_multibatch_test");
    if (!(rnnt_helper::is_close(scores_fwd[0], static_cast<float>(-log(0.363))) &&
          rnnt_helper::is_close(scores_fwd[1], static_cast<float>(-log(0.363))))) {
        std::cout << "Align-restricted multibatch test with 3 frame shift failed." << std::endl;
        status = false;
    }

    // Shift of 0 means sticking to alignments [0, 1, 0, 2] and [1, 2, 0, 0]
    workspace_manager.restrict_to_alignment(alignment_gpu, 0, 0);
    throw_on_error(rnnt_computer.cost(scores_fwd.data()), "Error: rnnt_computer forward in align_restrict_multibatch_test");
    if (!(rnnt_helper::is_close(scores_fwd[0], static_cast<float>(-log(0.072))) &&
          rnnt_helper::is_close(scores_fwd[1], static_cast<float>(-log(0.0672))))) {
        std::cout << "Align-restricted multibatch test with 0 frame shift failed." << std::endl;
        status = false;
    }

    // Shift of 1 means removing only path [1, 2, 0, 0] for the first alignment or paths [1, 0, 0, 2], [0, 1, 0, 2] and
    // [0, 0, 1, 2] for the second
    workspace_manager.restrict_to_alignment(alignment_gpu, 1, 0);
    throw_on_error(rnnt_computer.cost(scores_fwd.data()), "Error: rnnt_computer forward in align_restrict_multibatch_test");
    if (!(rnnt_helper::is_close(scores_fwd[0], static_cast<float>(-log(0.2958))) &&
          rnnt_helper::is_close(scores_fwd[1], static_cast<float>(-log(0.192))))) {
        std::cout << "Align-restricted multibatch test with 1 frame shift failed." << std::endl;
        status = false;
    }

    workspace_manager.free_workspace();

    cudaFree(logits_gpu);
    cudaFree(labels_gpu);
    cudaFree(lengths_gpu);
    cudaFree(label_lengths_gpu);

    cudaStreamDestroy(stream);

    return status;
}

int main() {
    std::cout << "Running gpu tests" << std::endl;

    bool status = true;
    status &= fwd_test();
    printf("finish fwd_test %d\n", status);
    status &= bwd_test();
    printf("finish bwd_test %d\n", status);
    status &= grads_test();
    printf("finish grads_test %d\n", status);
    status &= multibatch_test();
    printf("finish multibatch_test %d\n", status);
    status &= infnan_test();
    printf("finish infnan_test %d\n", status);
    status &= align_restrict_test();
    printf("finish align_restrict_test %d\n", status);
    status &= align_restrict_multibatch_test();
    printf("finish align_restrict_multibatch_test %d\n", status);

    if (status) {
        std::cout << "Tests pass" << std::endl;
        return 0;
    } else {
        std::cout << "Some or all tests fail" << std::endl;
        return 1;
    }
}
