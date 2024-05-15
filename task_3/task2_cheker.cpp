#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring> 

struct Task {
    std::string type;
    int num;
    double arg1;
    double arg2;
    double result;
};

int Sinus(const std::string& filename){
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка при открытии файла" << std::endl;
        return 1;
    }
    int count_err = 0;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);

        Task task;
        std::string taskStr;
        if (!(iss >> task.type >> task.num >> task.arg1 >> task.result)) {
            std::cerr << "Не получилось распарсить строку: " << line << std::endl;
        }

        double sinArg = std::sin(task.arg1);
        
        if (std::abs(sinArg - task.result) > 1e-5) {
            std::cout << "Task: " << task.num << " arg1 = " << task.arg1 << ", sin(arg1) = " << sinArg 
                      << ", Result = " << task.result << " (получившееся значение не соответствует исходному)" << std::endl;
            count_err++;
        }
    }
    file.close();
    return count_err;
}


int Sqrt(const std::string& filename){
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка при открытии файла" << std::endl;
        return 1;
    }
    int count_err = 0;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);

        Task task;
        std::string taskStr;
        if (!(iss >> task.type >> task.num >> task.arg1 >> task.result)) {
            std::cerr << "Не получилось распарсить строку: " << line << std::endl;
        }

        double sqrtArg = std::sqrt(task.arg1);
        
        if (std::abs(sqrtArg - task.result) > 1e-5) {
            std::cout << "Task: " << task.num << " arg1 = " << task.arg1 << ", sqrt(arg1) = " << sqrtArg 
                      << ", Result = " << task.result << " (получившееся значение не соответствует исходному)" << std::endl;
            count_err++;
        }
    }
    file.close();
    return count_err;
}


int Pow(const std::string& filename){
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка при открытии файла" << std::endl;
        return 1;
    }
    int count_err = 0;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);

        Task task;
        std::string taskStr;
        if (!(iss >> task.type >> task.num >> task.arg1 >> task.arg2 >> task.result)) {
            std::cerr << "Не получилось распарсить строку: " << line << std::endl;
        }

        double powArg = std::pow(task.arg1, task.arg2);
        
        if (std::abs(powArg - task.result) > 1e-5) {
            std::cout << "Task: " << task.num << " arg1 = " << task.arg1 << " arg2 = " << task.arg2 
                      << ", arg1^arg2 = " << powArg << ", Result = " << task.result 
                      << " (получившееся значение не соответствует исходному)" << std::endl;
            count_err++;
        }
    }
    file.close();
    return count_err;
}


int main(int argc, char *argv[]) {

    int check = 0;
    if (argc != 2 || (std::strcmp(argv[1], "Sinus.txt") != 0 && std::strcmp(argv[1], "Square.txt") != 0 && std::strcmp(argv[1], "Power.txt") != 0)){
        std::cerr << "Введите файл с разрешением .txt (Sinus.txt Square.txt Power.txt), который вы хотите проверить" << std::endl;
        return 1;
    }
    else {
        int err;
        if(std::strcmp(argv[1], "Sinus.txt") == 0){
            err = Sinus(argv[1]);
        }
        else if(std::strcmp(argv[1], "Square.txt") == 0){
            err = Sqrt(argv[1]);
        }
        else{
            err = Pow(argv[1]);
        }
        std::cout << "Файл проверен, количество ошибок:" << err << std::endl;
    }

    return 0;
}
