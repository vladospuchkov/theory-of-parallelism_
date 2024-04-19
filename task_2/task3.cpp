#include <iostream>
#include <cmath>
#include <memory>
#include <omp.h>
#include <string>

double eps = 0.00001;
double tau = 0.0001;
int potoks;
// using std::stoi;
// using std::endl;
// using std::unique_ptr;
using namespace std;

#ifdef TYPEDYNAMIC
#define SCHEDULE_TYPE dynamic
#endif
#ifdef TYPESTATIC
#define SCHEDULE_TYPE static
#endif
#ifdef TYPEGUIDED
#define SCHEDULE_TYPE guided
#endif

void initializeVectors(const int N, unique_ptr<double[]>& matrix_A,
                       unique_ptr<double[]>& vector_X, unique_ptr<double[]>& vector_b) {
    #pragma omp parallel for schedule(SCHEDULE_TYPE, int(N / (potoks * 2)))
    for (int i = 0; i < N; i++) {
        vector_X[i] = 0.0;
        vector_b[i] = static_cast<double>(N + 1);
        double bNormL2Local = 0.0;

        for (int j = 0; j < N; j++) {
            matrix_A[i * N + j] = (i == j) ? 2.0 : 1.0;
        }
    }
}

void calculateNormL2(const int N, const unique_ptr<double[]>& vector_b, double& bNormL2) {
    #pragma omp parallel for reduction(+:bNormL2) schedule(SCHEDULE_TYPE, int(N / (potoks * 2)))
    for (int i = 0; i < N; i++) {
        bNormL2 += vector_b[i] * vector_b[i];
    }

    bNormL2 = sqrt(bNormL2);
}

void calculateC(const int N, const unique_ptr<double[]>& matrix_A,
                const unique_ptr<double[]>& vector_X, const unique_ptr<double[]>& vector_b,
                unique_ptr<double[]>& vector_c) {
    #pragma omp parallel for schedule(SCHEDULE_TYPE, int(N / (potoks * 2)))
    for (int i = 0; i < N; i++) {
        vector_c[i] = 0.0;
        for (int j = 0; j < N; j++) {
            vector_c[i] += matrix_A[i * N + j] * vector_X[j];
        }
        vector_c[i] -= vector_b[i];
    }
}

double calculateSumOfSquaredNorms(const int N, const unique_ptr<double[]>& vector_c) {
    double sumloc = 0.0;
    #pragma omp parallel for reduction(+:sumloc) schedule(SCHEDULE_TYPE, int(N / (potoks * 2)))
    for (int i = 0; i < N; i++) {
        sumloc += vector_c[i] * vector_c[i];
    }
    return sumloc;
}

void updateVectorX(const int N, const double tau, const unique_ptr<double[]>& vector_c,
                   unique_ptr<double[]>& vector_X) {
    #pragma omp parallel for schedule(SCHEDULE_TYPE, int(N / (potoks * 2)))
    for (int i = 0; i < N; i++) {
        vector_X[i] -= tau * vector_c[i];
    }
}

void print_time_of_programm(double ans){
    cout << ans << endl;
}

int main(int argc, char const *argv[]) {
    if (argc < 3) {
        cout << " N= - размер матрицы и potoks= - количество потоков" << endl;
        return 0;
    }

    int N = stoi(argv[1]);
    potoks = stoi(argv[2]);

    omp_set_num_threads(potoks);

    unique_ptr<double[]> matrix_A(new double[N * N]);

    unique_ptr<double[]> vector_X(new double[N]);
    
    unique_ptr<double[]> vector_b(new double[N]);
    unique_ptr<double[]> vector_c(new double[N]);

    double sumOfSquaredNorms = 0.0;
    double prevNormSum = 0.0;
    double bNormL2 = 0.0;

    initializeVectors(N, matrix_A, vector_X, vector_b);
    calculateNormL2(N, vector_b, bNormL2);

    double start_of_programm = omp_get_wtime();
    while (true) {
        calculateC(N, matrix_A, vector_X, vector_b, vector_c);

        sumOfSquaredNorms = calculateSumOfSquaredNorms(N, vector_c);
        sumOfSquaredNorms = sqrt(sumOfSquaredNorms);

        if (bNormL2 == 0.0) {
            cout << "oшибка     -   норма bNormL2 равна нулю." << endl;
            break;
        }

        if (sumOfSquaredNorms / bNormL2 < eps) break;

        if (sumOfSquaredNorms > prevNormSum && prevNormSum != 0) {
            cout << "расхождение" << endl;
            break;
        }

        prevNormSum = sumOfSquaredNorms;
        sumOfSquaredNorms = 0;

        updateVectorX(N, tau, vector_c, vector_X);
    }

    double end_of_programm = omp_get_wtime();

    print_time_of_programm(end_of_programm - start_of_programm);
    
    return 0;
}
