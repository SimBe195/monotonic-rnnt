#ifndef MONOTONIC_RNNT_TEST_H
#define MONOTONIC_RNNT_TEST_H

#include <stdexcept>
#include <vector>
#include <limits>
#include <numeric>

#include <rnnt_entrypoint.h>

inline void throw_on_error(RNNTStatus status, const char *message) {
    if (status != RNNT_STATUS_SUCCESS) {
        throw std::runtime_error(message + (", status = " +
                                            std::string(rnntGetStatusString(status))));
    }
}

float *genActs(int size);

void genActs(std::vector<float> &arr);

std::vector<int> genLabels(int alphabet_size, int L);

float rel_diff(const std::vector<float> &grad,
               const std::vector<float> &num_grad) {
    float diff = 0.;
    float tot = 0.;
    for (size_t idx = 0; idx < grad.size(); ++idx) {
        diff += (grad[idx] - num_grad[idx]) * (grad[idx] - num_grad[idx]);
        tot += grad[idx] * grad[idx];
    }

    return diff / tot;
}

// Numerically stable softmax for a minibatch of 1
void softmax(const float *const acts,
             int alphabet_size, int T,
             float *probs, bool applylog) {

    for (int t = 0; t < T; ++t) {

        float max_activation =
                -std::numeric_limits<float>::infinity();

        for (int v = 0; v < alphabet_size; ++v) {
            max_activation =
                    std::max(max_activation, acts[t * alphabet_size + v]);
        }

        float denom = 0;
        for (int v = 0; v < alphabet_size; ++v) {
            denom += std::exp(acts[t * alphabet_size + v] - max_activation);
        }

        for (int v = 0; v < alphabet_size; ++v) {
            probs[t * alphabet_size + v] =
                    std::exp(acts[t * alphabet_size + v] - max_activation) / denom;
            if (applylog) {
                probs[t * alphabet_size + v] = std::log(probs[t * alphabet_size + v]);
            }
        }
    }
}

#endif //MONOTONIC_RNNT_TEST_H
