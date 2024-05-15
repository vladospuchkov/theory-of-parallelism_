#include <iostream>
#include <boost/program_options.hpp>
#include <thread>
#include <vector>
#include <chrono>

namespace opt = boost::program_options;

// lb - нижняя граница диапазона строк которые нужно вычислить
// ub -  верхняя граница диапазона строк которые нужно вычислить


void calculateRowProduct(const std::unique_ptr<double[]>& matrix, const std::unique_ptr<double[]>& vector, std::unique_ptr<double[]>& result, const int lb, const int ub, const int N)
{
    for (size_t row = lb; row < ub; row++)
    {
        for (size_t col = 0; col < N; col++)
        {
            result[row] += matrix[row * N + col] * vector[col];
        }
    }
}

void initializeVectors(std::unique_ptr<double[]>& matrix, const std::unique_ptr<double[]>& vector, std::unique_ptr<double[]>& result, const int lb, const int ub, const int N)
{
    for (size_t row = lb; row <= ub; row++)
    {
        vector[row] = static_cast<double>(row + 1);
        result[row] = 0.0;
        for (size_t col = 0; col < N; col++)
        {
            matrix[row * N + col] = (row == col) ? col : 1;
        }
    }
}


int main(){
    int threadCount = 40;
    // int N = 20000;
    int N = 40000;
    std::vector<std::thread> threads;
    int rowsPerThread = N / threadCount; // количество строк обрабатываемых каждым потоком

    std::unique_ptr<double[]> matrix(new double[N * N]);
    std::unique_ptr<double[]> vector(new double[N]);
    std::unique_ptr<double[]> result(new double[N]);

    for (size_t threadID = 0; threadID < threadCount; threadID++)
    {
        int lowerBound = threadID * rowsPerThread;
        int upperBound = (threadID == threadCount - 1) ? (N - 1) : (lowerBound + rowsPerThread - 1);

        threads.emplace_back(initializeVectors, std::ref(matrix), std::ref(vector), std::ref(result), lowerBound, upperBound, N);
    }

    for (auto &thread : threads)
    {
        thread.join();
    }

    threads.clear();

    const auto startTime = std::chrono::steady_clock::now();
    for (size_t threadID = 0; threadID < threadCount; threadID++)
    {
        int lowerBound = threadID * rowsPerThread;
        int upperBound = (threadID == threadCount - 1) ? (N - 1) : (lowerBound + rowsPerThread - 1);

        threads.emplace_back(calculateRowProduct, std::ref(matrix), std::ref(vector), std::ref(result), lowerBound, upperBound, N);
    }

    for (auto &thread : threads)
    {
        thread.join();
    }

    threads.clear();

    const auto endTime = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsedTime = endTime - startTime;
    std::cout << "Task completed. Total time: " << elapsedTime.count() << " seconds" << std::endl;

    return 0;
}
