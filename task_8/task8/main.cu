#include <iostream>
#include <boost/program_options.hpp> 
#include <cmath> 
#include <memory> 
#include <algorithm> 
#include <fstream> 
#include <iomanip> 
#include <chrono> 

namespace opt = boost::program_options; 

#include <cuda_runtime.h> 
#include <cub/cub.cuh> 


// cuda unique_ptr

template<typename T>
using cuda_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>; //умного указатель для CUDA

// new
template<typename T>
T* cuda_new(size_t size)
{
    T *d_ptr;
    cudaMalloc((void **)&d_ptr, sizeof(T) * size); // выделение памяти на устройстве
    return d_ptr;
}

// delete
template<typename T>
void cuda_delete(T *dev_ptr)
{
    cudaFree(dev_ptr); //восвобождение памяти на устройстве
}


//захват вычислительного графа используется для оптимизации и упрощения выполнения повторяющихся вычислений
cudaStream_t* cuda_new_stream()
{
    cudaStream_t* stream = new cudaStream_t; //создание нового CUDA потока
    cudaStreamCreate(stream); //инициализация потока
    return stream;
}


void cuda_delete_stream(cudaStream_t* stream)
{
    cudaStreamDestroy(*stream); // Уничтожение CUDA потока
    delete stream;
}



//CUDA-графы позволяют записывать последовательность операций и запускать их как единое целое,
//что может приводить к значительному улучшению производительности по сравнению 
//с традиционным подходом запуска отдельных CUDA-ядр

//CUDA-графы здесь используются для записи и последующего выполнения серии вычислительных задач
cudaGraph_t* cuda_new_graph()
{
    cudaGraph_t* graph = new cudaGraph_t; // Создание нового CUDA графа
    return graph;
}

void cuda_delete_graph(cudaGraph_t* graph)
{
    cudaGraphDestroy(*graph); // Уничтожение CUDA графа
    delete graph;
}



//создает и возвращает указатель на новый объект типа cudaGraphExec_t
//не обычный объект, который можно просто создать с помощью new. 
//Этот тип является handle (дескриптором), который используется CUDA API для управления исполняемыми графами
cudaGraphExec_t* cuda_new_graph_exec()
{
    cudaGraphExec_t* graphExec = new cudaGraphExec_t; // Создание нового исполняемого CUDA графа
    return graphExec;
}

void cuda_delete_graph_exec(cudaGraphExec_t* graphExec)
{
    cudaGraphExecDestroy(*graphExec); // Уничтожение исполняемого CUDA графа
    delete graphExec;
}

#define CHECK(call)                                                             \
    {                                                                           \
        const cudaError_t error = call;                                         \
        if (error != cudaSuccess)                                               \
        {                                                                       \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
            printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                            \
        }                                                                       \
    } //макрос для проверки ошибок CUDA вызовов

double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1)); // Функция линейной интерполяции
}

void initMatrix(std::unique_ptr<double[]> &arr ,int N){
    for (size_t i = 0; i < N*N-1; i++)
    {
        arr[i] = 0; // Инициализация массива нулями
    }

    arr[0] = 10.0;
    arr[N-1] = 20.0;
    arr[(N-1)*N + (N-1)] = 30.0;
    arr[(N-1)*N] = 20.0;
    // Инициализация граничных значений матрицы
    for (size_t i = 1; i < N-1; i++)
    {
        arr[0*N+i] = linearInterpolation(i,0.0,arr[0],N-1,arr[N-1]);
        arr[i*N+0] = linearInterpolation(i,0.0,arr[0],N-1,arr[(N-1)*N]);
        arr[i*N+(N-1)] = linearInterpolation(i,0.0,arr[N-1],N-1,arr[(N-1)*N + (N-1)]);
        arr[(N-1)*N+i] = linearInterpolation(i,0.0,arr[(N-1)*N],N-1,arr[(N-1)*N + (N-1)]);
    }
}

void saveMatrixToFile(const double* matrix, int N, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {

        std::cerr << "Не удалось открыть файл " << filename << " для записи" << std::endl;

        return;
    }

    //yстанавливаем ширину вывода для каждого элемента
    int fieldWidth = 10; //ширина поля вывода, можно настроить по вашему усмотрению

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            outputFile << std::setw(fieldWidth) << std::fixed << std::setprecision(4) << matrix[i * N + j];
        }
        outputFile << std::endl;
    }

    outputFile.close();
}

void swapMatrices(double* &prevmatrix, double* &curmatrix) {
    double* temp = prevmatrix;
    prevmatrix = curmatrix;
    curmatrix = temp; 
    // Функция для обмена указателей на матрицы
}
//  предназначена для выполнения одной итерации вычислений для каждого элемента матрицы, за исключением граничных элементов
//glabal чтобы позволяет выполнить параллельно на всех потоках
__global__ void computeOneIteration(double *prevmatrix, double *curmatrix, int size){
    //Эти строки кода вычисляют индексы i и j, которые соответствуют строке и столбцу элемента матрицы.
    // blockIdx и threadIdx — это встроенные переменные CUDA,
    // представляющие индексы блока и потока соответственно, а blockDim — размер блока.
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // чтобы не изменять границы
    //этот условный оператор проверяет, находится ли текущий элемент на границе матрицы. Если это так, элемент не будет обновлен.
    if (!(j == 0 || i == 0 || i >= size-1 || j >= size-1))
        curmatrix[i*size+j]  = 0.25 * (prevmatrix[i*size+j+1] + prevmatrix[i*size+j-1] + prevmatrix[(i-1)*size+j] + prevmatrix[(i+1)*size+j]); 
    // Вычисление нового значения элемента матрицы
}


//предназначена для вычисления разницы между соответствующими элементами двух матриц (prevmatrix и curmatrix) 
//и сохранения абсолютного значения этой разницы в матрице error.
//В результате, матрица error содержит абсолютные значения разниц внутренних элементов двух матриц
__global__ void matrixSub(double *prevmatrix, double *curmatrix, double *error, int size){
        //Эти строки кода вычисляют индексы i и j, которые соответствуют строке и столбцу элемента матрицы.
    // blockIdx и threadIdx — это встроенные переменные CUDA,
    // представляющие индексы блока и потока соответственно, а blockDim — размер блока.
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // чтобы не изменять границы
    if (!(j == 0 || i == 0 || i >= size-1 || j >= size-1))
        error[i*size + j] = fabs(curmatrix[i*size+j] - prevmatrix[i*size+j]); 
    // Вычисление разности между элементами двух матриц
}

int main(int argc, char const *argv[]){
    opt::options_description desc("опции");
    desc.add_options()
        ("accuracy", opt::value<double>()->default_value(1e-6), "точность") 
        ("cellsCount", opt::value<int>()->default_value(1024), "размер матрицы") 
        ("iterCount", opt::value<int>()->default_value(1000000), "количество операций") 
        ("device", opt::value<int>()->default_value(3), "номер видеокарты") 
        ("help", "помощь") 
    ;

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm); 

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    opt::notify(vm);

    int N = vm["cellsCount"].as<int>();
    double accuracy = vm["accuracy"].as<double>();
    int countIter = vm["iterCount"].as<int>();
    int deviceNum = vm["device"].as<int>();


    CHECK(cudaSetDevice(deviceNum)); // устройство CUDA

    cuda_unique_ptr<cudaStream_t> stream(cuda_new_stream(), cuda_delete_stream); //cоздание умного указателя на CUDA поток
    cuda_unique_ptr<cudaGraph_t> graph(cuda_new_graph(), cuda_delete_graph); //cоздание умного указателя на CUDA граф
    cuda_unique_ptr<cudaGraphExec_t> g_exec(cuda_new_graph_exec(), cuda_delete_graph_exec); //создание умного указателя на исполняемый CUDA граф

    double error = 1.0;
    int iter = 0;

    std::unique_ptr<double[]> A(new double[N*N]); //выделение памяти для матрицы A
    std::unique_ptr<double[]> Anew(new double[N*N]); //выделение памяти для матрицы Anew
    std::unique_ptr<double[]> B(new double[N*N]); //выделение памяти для матрицы B (ошибок)

    initMatrix(std::ref(A), N); 
    initMatrix(std::ref(Anew), N); 

    double* curmatrix = A.get();
    double* prevmatrix = Anew.get();
    double* error_matrix = B.get();

    cuda_unique_ptr<double> curmatrix_GPU_ptr(cuda_new<double>(N*N), cuda_delete<double>); //выделение памяти на устройстве для текущей матрицы
    cuda_unique_ptr<double> prevmatrix_GPU_ptr(cuda_new<double>(N*N), cuda_delete<double>); //выделение памяти на устройстве для предыдущей матрицы
    cuda_unique_ptr<double> error_gpu_ptr(cuda_new<double>(N*N), cuda_delete<double>); //выделение памяти на устройстве для матрицы ошибок
    cuda_unique_ptr<double> error_GPU_ptr(cuda_new<double>(1), cuda_delete<double>); //выделение памяти на устройстве для значения ошибки 1 чтобы могло пройти на первой итерации 

    double* curmatrix_GPU = curmatrix_GPU_ptr.get();
    double* prevmatrix_GPU = prevmatrix_GPU_ptr.get();
    double* error_gpu = error_gpu_ptr.get();
    double* error_GPU = error_GPU_ptr.get();

    CHECK(cudaMemcpy(curmatrix_GPU, curmatrix, N*N*sizeof(double), cudaMemcpyHostToDevice)); //копирование данных текущей матрицы на устройство
    CHECK(cudaMemcpy(prevmatrix_GPU, prevmatrix, N*N*sizeof(double), cudaMemcpyHostToDevice)); //копирование данных предыдущей матрицы на устройство
    CHECK(cudaMemcpy(error_gpu, error_matrix, N*N*sizeof(double), cudaMemcpyHostToDevice)); //копирование данных матрицы ошибок на устройство

    size_t tmp_size = 0;
    double* tmp = nullptr;
    
    //yказатель на временное хранилище на устройстве. Это хранилище требуется для промежуточных шагов редукции
    //ссылка на переменную, содержащую размер (в байтах) требуемого временного хранилища
    //итератор на входной массив, в котором выполняется редукция
    //4.Итератор на выходное значение, куда будет записан результат редукции
    //Количество элементов во входном массиве.
    // Поток CUDA, в котором будет выполняться операция
    
    //Этот вызов используется для получения размера необходимого временного хранилища. Так как tmp равно nullptr, 
    //функция не выполняет реальную операцию редукции, а только определяет, сколько памяти потребуется для выполнения этой операции
    cub::DeviceReduce::Max(tmp, tmp_size, prevmatrix_GPU, error_GPU, N*N); //предварительное вычисление необходимого размера временной памяти для редукции

    cuda_unique_ptr<double> tmp_ptr(cuda_new<double>(tmp_size), cuda_delete<double>); // Выделение временной памяти
    tmp = tmp_ptr.get();
//----------------------------------------------------------------------------------------------------------------------------------------------------------------

//Блоки группируют потоки, чтобы обеспечить совместный доступ к разделяемой памяти и упростить управление ими.
    dim3 threads_in_block = dim3(32, 32); //определение числа потоков в блоке
    dim3 blocks_in_grid((N + threads_in_block.x - 1) / threads_in_block.x, (N + threads_in_block.y - 1) / threads_in_block.y); //лпределение числа блоков в сетке



    // начало записи вычислительного графа

    //позволяет записывать последовательность операций CUDA в граф и затем выполнять этот граф как одно целое
    //вз cudaError_t
    cudaStreamBeginCapture(*stream, cudaStreamCaptureModeGlobal); //начало захвата вычислительного графа

    for(size_t i = 0; i < 999; i++){
        computeOneIteration<<<blocks_in_grid, threads_in_block, 0, *stream>>>(prevmatrix_GPU, curmatrix_GPU, N); //запуск одной итерации вычислений
        swapMatrices(prevmatrix_GPU, curmatrix_GPU); //обмен указателей на матрицы
    }


    //глобальные функции __glabal__
    computeOneIteration<<<blocks_in_grid, threads_in_block, 0, *stream>>>(prevmatrix_GPU, curmatrix_GPU, N); //выполнение последней итерации вычислений
    matrixSub<<<blocks_in_grid, threads_in_block, 0, *stream>>>(prevmatrix_GPU, curmatrix_GPU, error_gpu, N); //вычисление разности между матрицами

    cub::DeviceReduce::Max(tmp, tmp_size, error_gpu, error_GPU, N*N, *stream); // вычисление максимальной ошибки
    cudaStreamEndCapture(*stream, graph.get()); //конец захвата вычислительного графа

    cudaGraphInstantiate(g_exec.get(), *graph, NULL, NULL, 0); //cоздание исполняемого графа

    auto start = std::chrono::high_resolution_clock::now(); 
    while(error > accuracy && iter < countIter){
        cudaGraphLaunch(*g_exec, *stream); //запуск исполняемого графа
        cudaMemcpy(&error, error_GPU, 1*sizeof(double), cudaMemcpyDeviceToHost); // Копирование значения ошибки на хост
        iter += 1000;
        std::cout << "Итерация: " << iter << ' ' << "ошибка: " << error << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto time_s = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); 
    std::cout << "Время: " << time_s << " мс, Ошибка: " << error << ", Итерации: " << iter << std::endl; // Вывод результата


    CHECK(cudaMemcpy(prevmatrix, prevmatrix_GPU, sizeof(double)*N*N, cudaMemcpyDeviceToHost)); //копирование данных предыдущей матрицы на хост
    CHECK(cudaMemcpy(error_matrix, error_gpu, sizeof(double)*N*N, cudaMemcpyDeviceToHost)); //копирование данных матрицы ошибок на хост
    CHECK(cudaMemcpy(curmatrix, curmatrix_GPU, sizeof(double)*N*N, cudaMemcpyDeviceToHost)); //копирование данных текущей матрицы на хост

    if (N == 13 || N == 10){
        
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                std::cout << A[i*N+j] << ' ';
            }
            std::cout << std::endl;
        }
        std::cout<< '\n' << std::endl;
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                std::cout << Anew[i*N+j] << ' ';
            }
            std::cout << std::endl;
        }

    }
    // saveMatrixToFile(curmatrix, N , "matrix.txt");

    saveMatrixToFile(curmatrix, N, "matrix.txt"); 

    return 0;
}
