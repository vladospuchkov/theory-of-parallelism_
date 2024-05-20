#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring> 


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

        std::string taskStr, argStr, resultStr;
        double arg, result, num;

        if (!(iss >> taskStr >> num >> argStr >> arg >> resultStr >> result)) {
            std::cerr << "Не получилось распарсить строку: " << line << std::endl;
        }

        double sinArg = std::sin(arg);
        
        if (std::abs(sinArg - result) > 1e-5) {
            std::cout << "Task: "<< num << " arg1 = " << arg << ", sin(arg1) = " << sinArg << ", Result = " << result << 
            " (получившееся значение не соответствует исходному)" << std::endl;
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

        std::string taskStr, argStr, resultStr;
        double arg, result, num;

        if (!(iss >> taskStr >> num >> argStr >> arg >> resultStr >> result)) {
            std::cerr << "Не получилось распарсить строку: " << line << std::endl;
        }

        double sqrtArg = std::sqrt(arg);
        
        if (std::abs(sqrtArg - result) > 1e-5) {
            std::cout << "Task: "<< num << " arg1 = " << arg << ", sin(arg1) = " << sqrtArg << ", Result = " << result << 
            " (получившееся значение не соответствует исходному)" << std::endl;
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

        std::string taskStr, argStr1, argStr2, resultStr;
        double arg1, arg2, result, num;

        if (!(iss >> taskStr >> num >> argStr1 >> arg1 >> argStr2 >> arg2 >> resultStr >> result)) {
            std::cerr << "Не получилось распарсить строку: " << line << std::endl;
        }

        double powArg = std::pow(arg1, arg2);
        
        if (std::abs(powArg - result) > 1e-5) {
            std::cout << "Task: "<< num << " arg1 = " << arg1 << " arg2 = " << arg2 << ", arg1^arg2 = " << powArg << ", Result = " << result << 
            " (получившееся значение не соответствует исходному)" << std::endl;
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
