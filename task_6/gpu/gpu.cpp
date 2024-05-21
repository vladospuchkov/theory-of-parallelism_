#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
    #include <openacc.h>
namespace opt = boost::program_options;

// функция для линейной интерполяции
double linearInterpolate(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
}

// инициализация матрицы
void initializeMatrix(std::unique_ptr<double[]> &matrix, int size) {
    matrix[0] = 10.0;
    matrix[size - 1] = 20.0;
    matrix[(size - 1) * size + (size - 1)] = 30.0;
    matrix[(size - 1) * size] = 20.0;

    for (size_t i = 1; i < size - 1; ++i) {
        matrix[i] = linearInterpolate(i, 0.0, matrix[0], size - 1, matrix[size - 1]);
        matrix[i * size] = linearInterpolate(i, 0.0, matrix[0], size - 1, matrix[(size - 1) * size]);
        matrix[i * size + (size - 1)] = linearInterpolate(i, 0.0, matrix[size - 1], size - 1, matrix[(size - 1) * size + (size - 1)]);
        matrix[(size - 1) * size + i] = linearInterpolate(i, 0.0, matrix[(size - 1) * size], size - 1, matrix[(size - 1) * size + (size - 1)]);
    }
}


void saveMatrixToFile(const double* matrix, int size, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Не удалось открыть файл " << filename << " для записи." << std::endl;
        return;
    }

    int fieldWidth = 10;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            outputFile << std::setw(fieldWidth) << std::fixed << std::setprecision(4) << matrix[i * size + j];
        }
        outputFile << std::endl;
    }

    outputFile.close();
}

int main(int argc, char const *argv[]) {
    opt::options_description desc("Options");
    desc.add_options()
        ("accuracy", opt::value<double>()->default_value(1e-6), "Точность")
        ("size", opt::value<int>()->default_value(1024), "Размер матрицы")
        ("iterations", opt::value<int>()->default_value(1000000), "Количество итераций")
        ("help", "Help message");

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    int size = vm["size"].as<int>();
    double accuracy = vm["accuracy"].as<double>();
    int maxIterations = vm["iterations"].as<int>();

    acc_set_device_num(3, acc_device_nvidia);

    double error = 1.0;
    int iteration = 0;

    std::unique_ptr<double[]> matrix(new double[size * size]);
    std::unique_ptr<double[]> newMatrix(new double[size * size]);

    initializeMatrix(matrix, size);
    initializeMatrix(newMatrix, size);

    auto start = std::chrono::high_resolution_clock::now();
    double* currentMatrix = matrix.get();
    double* previousMatrix = newMatrix.get();

    #pragma acc data copyin(error, previousMatrix[0:size * size], currentMatrix[0:size * size])
    {
        while (iteration < maxIterations && error > accuracy) {
            #pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(1024) present(currentMatrix, previousMatrix)
            for (size_t i = 1; i < size - 1; ++i) {
                for (size_t j = 1; j < size - 1; ++j) {
                    currentMatrix[i * size + j] = 0.25 * (previousMatrix[i * size + j + 1] + previousMatrix[i * size + j - 1] + previousMatrix[(i - 1) * size + j] + previousMatrix[(i + 1) * size + j]);
                }
            }

            if ((iteration + 1) % 10000 == 0) {
                error = 0.0;
                #pragma acc update device(error)
                #pragma acc parallel loop independent collapse(2) vector vector_length(1024) gang num_gangs(256) reduction(max:error) present(currentMatrix, previousMatrix)
                for (size_t i = 1; i < size - 1; ++i) {
                    for (size_t j = 1; j < size - 1; ++j) {
                        error = fmax(error, fabs(currentMatrix[i * size + j] - previousMatrix[i * size + j]));
                    }
                }

                #pragma acc update self(error)
                std::cout << "Итерация: " << iteration + 1 << " ошибка: " << error << std::endl;
            }

            std::swap(previousMatrix, currentMatrix);
            ++iteration;
        }
        #pragma acc update self(currentMatrix[0:size * size])
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    std::cout << "Время: " << timeMs << " мс, Ошибка: " << error << ", Итерации: " << iteration << std::endl;

    if (size == 13 || size == 10) {
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                std::cout << matrix[i * size + j] << ' ';
            }
            std::cout << std::endl;
        }
    }

    saveMatrixToFile(currentMatrix, size, "matrix.txt");

    matrix = nullptr;
    newMatrix = nullptr;
    return 0;
}
