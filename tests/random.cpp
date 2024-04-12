#include <random>
#include <vector>

float *genActs(int size) {
    auto *arr = new float[size];
    std::mt19937 gen(0);
    std::uniform_real_distribution<> dis(0, 1);
    for (int i = 0; i < size; ++i) {
        arr[i] = static_cast<float>(dis(gen));
    }
    return arr;
}

void genActs(std::vector<float> &arr) {
    std::mt19937 gen(0);
    std::uniform_real_distribution<> dis(0, 1);
    for (float &i : arr) {
        i = static_cast<float>(dis(gen));
    }
}

std::vector<int> genLabels(int V, int S) {
    std::vector<int> label(S);

    std::mt19937 gen(1);
    std::uniform_int_distribution<> dis(1, V - 1);

    for (int i = 0; i < S; ++i) {
        label[i] = dis(gen);
    }
    // guarantee repeats for testing
    if (S >= 3) {
        label[S / 2] = label[S / 2 + 1];
        label[S / 2 - 1] = label[S / 2];
    }
    return label;
}
