/*
Name: Harsh Raj
Experiment: OpenMP vs CUDA Dot Product
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include <cuda.h>

#define N 100000000
#define THREADS_PER_BLOCK 256
#define BLOCKS 1024

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
// CUDA Kernel
// ==============================
__global__
void dotProductKernel(double* A, double* B, double* partial, size_t n) {

    __shared__ double cache[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double temp = 0.0;

    while (index < n) {
        temp += A[index] * B[index];
        index += stride;
    }

    cache[tid] = temp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            cache[tid] += cache[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial[blockIdx.x] = cache[0];
}

// ==============================
// CUDA Wrapper
// ==============================
double dotProductCUDA(double* A, double* B, size_t n, float &kernelTime) {

    double *d_A, *d_B, *d_partial;
    size_t size = n * sizeof(double);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_partial, BLOCKS * sizeof(double));

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    dotProductKernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_A, d_B, d_partial, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&kernelTime, start, stop);

    std::vector<double> h_partial(BLOCKS);
    cudaMemcpy(h_partial.data(), d_partial, BLOCKS * sizeof(double), cudaMemcpyDeviceToHost);

    double finalSum = 0.0;
    for (int i = 0; i < BLOCKS; i++)
        finalSum += h_partial[i];

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partial);

    return finalSum;
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

    // OpenMP timing
    auto t0 = std::chrono::high_resolution_clock::now();
    double ompResult = dotProductOMP(A, B, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> ompTime = t1 - t0;

    // CUDA timing (total)
    float kernelTime = 0.0;

    auto t2 = std::chrono::high_resolution_clock::now();
    double cudaResult = dotProductCUDA(A, B, N, kernelTime);
    auto t3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cudaTotalTime = t3 - t2;

    std::cout << "\nOpenMP Result: " << ompResult;
    std::cout << "\nCUDA Result:   " << cudaResult;

    std::cout << "\n\nOpenMP Time: " << ompTime.count() << " sec";
    std::cout << "\nCUDA Total Time: " << cudaTotalTime.count() << " sec";
    std::cout << "\nCUDA Kernel Time: " << kernelTime / 1000.0 << " sec";

    std::cout << "\n\nSpeedup (OMP / CUDA total): "
              << ompTime.count() / cudaTotalTime.count() << "x\n";

    delete[] A;
    delete[] B;

    return 0;
}
