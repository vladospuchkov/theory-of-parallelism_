#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <sstream>
#include <boost/program_options.hpp>
namespace opt = boost::program_options;

// Функция для чтения матрицы из файла
std::unique_ptr<double[]> readMatrixFromFile(const std::string& filename, int& rows, int& cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return nullptr;
    }

    

    // Выделение памяти для матрицы
    std::unique_ptr<double[]> matrix(new double[rows * cols]);

    // Чтение значений матрицы
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> matrix[i * cols + j];
        }
    }

    file.close();
    return matrix;
}

int main(int argc, char const *argv[]) {

    opt::options_description desc("опции");
    desc.add_options()
        
        ("cellsCount",opt::value<int>()->default_value(256),"размер матрицы")
        
        ("help","помощь")
    ;

    opt::variables_map vm;

    opt::store(opt::parse_command_line(argc, argv, desc), vm);

    opt::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    
    // и это всё было только ради того чтобы спарсить аргументы.......

    int N = vm["cellsCount"].as<int>();


    std::string filename = "matrix.txt";

    // Чтение матрицы из файла
    std::unique_ptr<double[]> prevmatrix = readMatrixFromFile(filename, N, N);
    double error = 0;
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            if (!(j == 0 || i == 0 || i >= N-1 || j >= N-1)){
                double temp  = fabs(prevmatrix[i*N+j] - (0.25 * (prevmatrix[i*N+j+1] + prevmatrix[i*N+j-1] + prevmatrix[(i-1)*N+j] + prevmatrix[(i+1)*N+j])));
                error = fmax(error,temp);
            }
        }
        
    }
    

    if (prevmatrix == nullptr) {
        return -1;
    }
    std::cout << "ошибка: " << error <<std::endl;

    

    return 0;
}
