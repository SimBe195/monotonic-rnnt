#ifndef MONOTONIC_RNNT_TEST_H
#define MONOTONIC_RNNT_TEST_H

#include <stdexcept>
#include <vector>
#include <limits>
#include <numeric>

#include "status.h"

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

#endif //MONOTONIC_RNNT_TEST_H
