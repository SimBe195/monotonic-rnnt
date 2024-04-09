#include <cmath>
#include <random>
#include <tuple>
#include <vector>
#include <algorithm>

#include <iostream>

#include "cpu_rnnt.h"

#include "test.h"
#include "options.h"
#include "rnnt_helper.h"

bool fwd_test() {
    // Similar to example in README file
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

    std::vector<float> logits(probs.size());
    std::transform(probs.begin(), probs.end(), logits.begin(), log);

    RNNTOptions options{};
    options.blank_label = 0;
    options.num_threads = 1;

    size_t cpu_alloc_bytes;
    RNNTWorkspaceManager<float> workspace_manager(logits.data(), labels.data(), B, lengths.data(),
                                                  label_lengths.data(), V);
    throw_on_error(workspace_manager.get_workspace_size(&cpu_alloc_bytes), "Error: get_workspace_size in fwd_test");

    void *rnnt_cpu_workspace = malloc(cpu_alloc_bytes);
    workspace_manager.set_workspace(rnnt_cpu_workspace);

    float score_fwd;
    CpuRNNTComputer<float> rnnt_computer(workspace_manager, options.blank_label, options.num_threads);

    throw_on_error(rnnt_computer.cost(&score_fwd),
                   "Error: compute_rnnt_loss forward in fwd_test");

    free(rnnt_cpu_workspace);

    return rnnt_helper::is_close(score_fwd, static_cast<float>(-log(0.363)));
}

bool bwd_test() {
    // Similar to example in README file
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

    std::vector<float> logits(probs.size());
    std::transform(probs.begin(), probs.end(), logits.begin(), log);

    RNNTOptions options{};
    options.blank_label = 0;
    options.num_threads = 1;

    size_t cpu_alloc_bytes;
    RNNTWorkspaceManager<float> workspace_manager(logits.data(), labels.data(), B, lengths.data(),
                                                  label_lengths.data(), V);
    throw_on_error(workspace_manager.get_workspace_size(&cpu_alloc_bytes), "Error: get_workspace_size in bwd_test");

    void *rnnt_cpu_workspace = malloc(cpu_alloc_bytes);
    workspace_manager.set_workspace(rnnt_cpu_workspace);

    float score_fwd;
    CpuRNNTComputer<float> rnnt_computer(workspace_manager, options.blank_label, options.num_threads);
    throw_on_error(rnnt_computer.cost(&score_fwd),
                   "Error: compute_rnnt_loss forward in bwd_test");

    std::vector<float> grads(logits.size());
    float score_bwd;
    throw_on_error(rnnt_computer.cost_and_grad(&score_bwd, grads.data()),
                   "Error: compute_rnnt_loss forward+backward in bwd_test");

    free(rnnt_cpu_workspace);

    return rnnt_helper::is_close(score_fwd, score_bwd);
}

bool grads_test() {
    // Similar to example in README file
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

    std::vector<float> logits(probs.size());
    std::transform(probs.begin(), probs.end(), logits.begin(), log);

    RNNTOptions options{};
    options.blank_label = 0;
    options.num_threads = 1;

    size_t cpu_alloc_bytes;
    RNNTWorkspaceManager<float> workspace_manager(logits.data(), labels.data(), B, lengths.data(),
                                                  label_lengths.data(), V);
    throw_on_error(workspace_manager.get_workspace_size(&cpu_alloc_bytes), "Error: get_workspace_size in grads_test");

    void *rnnt_cpu_workspace = malloc(cpu_alloc_bytes);
    workspace_manager.set_workspace(rnnt_cpu_workspace);

    CpuRNNTComputer<float> rnnt_computer(workspace_manager, options.blank_label, options.num_threads);
    std::vector<float> grads(logits.size());
    float score;
    throw_on_error(rnnt_computer.cost_and_grad(&score, grads.data()),
                   "Error: compute_rnnt_loss forward+backward in grads_test");

    free(rnnt_cpu_workspace);

    std::vector<float> expected_grads = {
            // t = 0
            0.04, -0.14, 0.1,  // s = 0
            0.0, 0.0, 0.0,  // s = 1
            0.0, 0.0, 0.0,  // s = 2

            // t = 1
            0.13, -0.19, 0.06,  // s = 0
            -0.04, 0.04, -0.01,  // s = 1
            0.0, 0.0, 0.0,  // s = 2

            // t = 2
            0.06, -0.1, 0.04,  // s = 0
            0.01, 0.07, -0.08,  // s = 1
            -0.06, 0.04, 0.02,  // s = 2

            // t = 3
            0.0, 0.0, 0.0,  // s = 0
            0.14, 0.05, -0.19,  // s = 1
            -0.11, 0.05, 0.05   // s = 2
    };

    bool grads_close = true;

    for (size_t idx = 0ul; idx < expected_grads.size(); ++idx) {
        grads_close &= std::abs(expected_grads[idx] - grads[idx]) < 1e-02;
    }

    return grads_close;
}

bool multibatch_test() {
    const int B = 2;
    const int V = 3;

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
    std::transform(probs.begin(), probs.end(), logits.begin(), log);

    RNNTOptions options{};
    options.blank_label = 0;
    options.num_threads = 1;

    size_t cpu_alloc_bytes;
    RNNTWorkspaceManager<float> workspace_manager(logits.data(), labels.data(), B, lengths.data(),
                                                  label_lengths.data(), V);
    throw_on_error(workspace_manager.get_workspace_size(&cpu_alloc_bytes),
                   "Error: get_workspace_size in multibatch_test");

    void *rnnt_cpu_workspace = malloc(cpu_alloc_bytes);
    workspace_manager.set_workspace(rnnt_cpu_workspace);

    CpuRNNTComputer<float> rnnt_computer(workspace_manager, options.blank_label, options.num_threads);
    std::vector<float> scores_fwd(B);
    throw_on_error(rnnt_computer.cost(scores_fwd.data()),
                   "Error: compute_rnnt_loss forward in multibatch_test");

    std::vector<float> grads(logits.size());
    std::vector<float> scores_bwd(B);
    throw_on_error(rnnt_computer.cost_and_grad(scores_bwd.data(), grads.data()),
                   "Error: compute_rnnt_loss forward+backward in multibatch_test");

    free(rnnt_cpu_workspace);

    std::vector<float> expected_grads = {
            // b = 0
            // t = 0
            -0.02, -0.08, 0.1,  // s = 0
            0.0, 0.0, 0.0,  // s = 1

            // t = 1
            0.31, -0.37, 0.06,  // s = 0
            -0.19, 0.04, 0.15,  // s = 1

            // b = 1
            // t = 0
            0.04, -0.14, 0.1,  // s = 0
            0.0, 0.0, 0.0,  // s = 1
            0.0, 0.0, 0.0,  // s = 2

            // t = 1
            0.13, -0.19, 0.06,  // s = 0
            -0.04, 0.04, -0.01,  // s = 1
            0.0, 0.0, 0.0,  // s = 2

            // t = 2
            0.06, -0.1, 0.04,  // s = 0
            0.01, 0.07, -0.08,  // s = 1
            -0.06, 0.04, 0.02,  // s = 2

            // t = 3
            0.0, 0.0, 0.0,  // s = 0
            0.14, 0.05, -0.19,  // s = 1
            -0.11, 0.05, 0.05   // s = 2
    };

    bool grads_close = true;

    for (size_t idx = 0ul; idx < expected_grads.size(); ++idx) {
        grads_close &= std::abs(expected_grads[idx] - grads[idx]) < 1e-02;
    }

    return rnnt_helper::is_close(scores_fwd[0], static_cast<float>(-log(0.39)))
           && rnnt_helper::is_close(scores_fwd[1], static_cast<float>(-log(0.363)))
           && rnnt_helper::is_close(scores_fwd[0], scores_bwd[0])
           && rnnt_helper::is_close(scores_fwd[1], scores_bwd[1])
           && grads_close;
}

bool infnan_test() {
    const int B = 1;
    const int T = 50;
    const int S = 10;
    const int V = 15;

    std::vector<int> labels = genLabels(V, S);
    std::vector<int> label_lengths = {S};

    std::vector<float> acts(B * T * (S + 1) * V);
    genActs(acts);

    std::vector<int> lengths = {T};

    std::vector<float> grads(acts.size());

    float cost;

    RNNTOptions options{};
    options.num_threads = 1;

    size_t cpu_alloc_bytes;
    RNNTWorkspaceManager<float> workspace_manager(acts.data(), labels.data(), B, lengths.data(),
                                                  label_lengths.data(), V);
    throw_on_error(workspace_manager.get_workspace_size(&cpu_alloc_bytes), "Error: get_workspace_size in infnan_test");

    void *rnnt_cpu_workspace = malloc(cpu_alloc_bytes);
    workspace_manager.set_workspace(rnnt_cpu_workspace);

    CpuRNNTComputer<float> rnnt_computer(workspace_manager, options.blank_label, options.num_threads);

    throw_on_error(rnnt_computer.cost_and_grad(&cost, grads.data()),
                   "Error: compute_rnnt_loss forward in infnan_test");

    free(rnnt_cpu_workspace);

    bool status = true;
    status &= !std::isinf(cost);
    status &= !std::isnan(cost);

    for (auto grad: grads) {
        status &= !std::isinf(grad);
        status &= !std::isnan(grad);
    }

    return status;
}

int main() {
    std::cout << "Running CPU tests" << std::endl;

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

    if (status) {
        std::cout << "Tests pass" << std::endl;
        return 0;
    } else {
        std::cout << "Some or all tests fail" << std::endl;
        return 1;
    }
}