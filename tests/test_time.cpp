#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cpu_rnnt.h"
#include "cpu_workspace_manager.h"
#include "test.h"

bool run_test(int B, int T, int S, int V, int num_threads) {
    int len = B * T * (S + 1) * V;
    std::vector<float> acts(len);
    genActs(acts);

    std::vector<int> lengths;
    std::vector<int> label_lengths;
    std::vector<int> labels = genLabels(V, S * T);

    for (int mb = 0; mb < B; ++mb) {
        lengths.push_back(T);
        label_lengths.push_back(S);
    }

    std::vector<float> costs(B);
    std::vector<float> grads(acts.size());

    CpuRNNTWorkspaceManager<float> workspace_manager(acts.data(), labels.data(), B, lengths.data(), labels.data(), V);
    throw_on_error(workspace_manager.create_workspace(), "Error: get_workspace_size in run_test");

    CpuRNNTComputer<float> rnnt_computer(workspace_manager, 0, num_threads);

    std::vector<float> time;
    for (int i = 0; i < 10; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        throw_on_error(rnnt_computer.cost_and_grad(costs.data(), grads.data()), "Error: compute_rnnt_loss in run_test");
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float> elapsed = end - start;
        time.push_back(elapsed.count() * 1000);
        printf("compute_rnnt_loss elapsed time: %.2f ms\n", elapsed.count() * 1000);
    }

    workspace_manager.free_workspace();

    float sum = 0;
    for (float i : time) {
        sum += i;
    }
    sum /= static_cast<float>(time.size());

    float variance = 0;
    for (float i : time) {
        variance += (i - sum) * (i - sum);
    }
    variance /= static_cast<float>(time.size());

    printf("Average time over %zu computations: %.2f ms, variance: %.2f\n", time.size(), sum, variance);

    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Arguments: <Batch size> <Time step> <Label length> <Alphabet size>\n";
        return 1;
    }

    int B = atoi(argv[1]);
    int T = atoi(argv[2]);
    int S = atoi(argv[3]);
    int V = atoi(argv[4]);
    printf("Arguments:\nBatch size: %d\nTime step: %d\nLabel length: %d\nAlphabet size: %d\n", B, T, S, V);

    int num_threads = 1;
    if (argc >= 6) {
        num_threads = atoi(argv[5]);
        printf("Num threads: %d\n", num_threads);
    }

    run_test(B, T, S, V, num_threads);
}
