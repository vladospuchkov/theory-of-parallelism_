#include <stdio.h>
#include <omp.h>
#include <stdlib.h>


int print_time_of_programm(double ans){
    printf("Время выполнения программы = %.16g\n", ans);
}

int main(int argc, char const *argv[]){   

        int potoks = atoi(argv[1]);
        int m  = atoi(argv[2]);
        int n = m;
        
        double *a,*b,*c;
        a = (double *)malloc(sizeof(*a)*m*n);
        b = (double *)malloc(sizeof(*b)*n);
        c = (double *)malloc(sizeof(*c)*m);
        
        #pragma omp parallel num_threads(potoks)
        {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();

        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);

        for (int j = lb; j < ub; j++){ 
            b[j] = j;
                }
        for (int i = lb; i < ub; i++) {
            for (int j = 0; j < n; j++){
                    a[i * n + j] = i + j;
            }

        }
        }
        
        double start_of_programm = omp_get_wtime( );

        #pragma omp parallel num_threads(potoks)
        {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();

        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        
        for (int i = lb; i < ub; i++) {
            c[i] = 0.0;
            for (int j = 0; j < n; j++){
                c[i] += a[i * n + j] * b[j];
            }
        }

        }
        double end_of_programm = omp_get_wtime( );

        print_time_of_programm(end_of_programm - start_of_programm);

        free(a);
        free(b);
        free(c);
    
    
    return 0;
}