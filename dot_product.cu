/*
Name: Harsh Raj
Experiment: OpenMP vs CUDA Dot Product
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>

#define N 100000000   // 100 Million elements

// ==============================
// OpenMP Dot Product
// ==============================
double dotProductOMP(double* A, double* B, size_t n) {
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < n; i++) {
        sum += A[i] * B[i];
    }

    return sum;
}

// ==============================
// MAIN
// ==============================
int main() {

    std::cout << "Generating vectors...\n";

    double* A = new double[N];
    double* B = new double[N];

    std::mt19937_64 rng(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < N; i++) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double result = dotProductOMP(A, B, N);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time = t1 - t0;

    std::cout << "OpenMP Result: " << result << "\n";
    std::cout << "OpenMP Time: " << time.count() << " sec\n";

    delete[] A;
    delete[] B;

    return 0;
}
