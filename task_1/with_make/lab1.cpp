#include <iostream>
#include <cmath>




#ifdef Double
using Type_Array = double;
#else
#ifdef Float
using Type_Array = float;
#endif
#endif



// const int SIZE = 10000000;
const int SIZE = 1E7;

int main() {
    Type_Array* array = new Type_Array[SIZE];
    // double period = 2 * M_PI / SIZE;
    
    for (int i = 0; i < SIZE; i++) {
        
        Type_Array x = static_cast<Type_Array> (i) / SIZE;
        array[i] = static_cast<Type_Array>(sin(2 * M_PI *x));
    }

    Type_Array sum = 0;
    for (int i = 0; i < SIZE; i++)
        sum += array[i];


    std::cout << sum << std::endl;

    return 0;
}


