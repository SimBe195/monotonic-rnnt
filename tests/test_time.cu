#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "gpu_rnnt.h"
#include "gpu_workspace_manager.h"
#include "test.h"

template <typename T>
void vector_to_gpu(T*& gpu_space, std::vector<T>& vec, cudaStream_t& stream) {
    cudaMalloc(&gpu_space, vec.size() * sizeof(T));
    cudaMemcpyAsync(gpu_space, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice, stream);
}

bool run_test(int B, int T, int S, int V) {
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

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float* acts_gpu;
    vector_to_gpu<float>(acts_gpu, acts, stream);
    float* grads_gpu;
    cudaMalloc(&grads_gpu, len * sizeof(float));
    int* labels_gpu;
    vector_to_gpu(labels_gpu, labels, stream);
    int* lengths_gpu;
    vector_to_gpu(lengths_gpu, lengths, stream);
    int* label_lengths_gpu;
    vector_to_gpu(label_lengths_gpu, label_lengths, stream);

    std::vector<float> time;
    for (int i = 0; i < 10; ++i) {
        GpuRNNTWorkspaceManager<float> workspace_manager(acts_gpu, labels_gpu, B, lengths_gpu, label_lengths_gpu, V);
        throw_on_error(workspace_manager.create_workspace(stream), "Error: get_workspace_size in run_test");

        GpuRNNTComputer<float> rnnt_computer(workspace_manager, 0, stream);

        auto start = std::chrono::high_resolution_clock::now();
        throw_on_error(rnnt_computer.cost_and_grad(costs.data(), grads_gpu), "Error: rnnt_computer in run_test");
        auto end = std::chrono::high_resolution_clock::now();

        workspace_manager.free_workspace();

        std::chrono::duration<float> elapsed = end - start;
        time.push_back(elapsed.count() * 1000);
        printf("compute_rnnt_loss elapsed time: %.2f ms\n", elapsed.count() * 1000);
    }

    cudaFree(acts_gpu);
    cudaFree(grads_gpu);
    cudaFree(labels_gpu);
    cudaFree(lengths_gpu);
    cudaFree(label_lengths_gpu);

    float sum = 0;
    for (float t : time) {
        sum += t;
    }
    sum /= static_cast<float>(time.size());

    float variance = 0;
    for (float t : time) {
        variance += (t - sum) * (t - sum);
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

    run_test(B, T, S, V);
}
