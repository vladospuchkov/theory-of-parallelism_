lib = -fopenmp

task1: task1.c
		gcc -o $@ task1.c $(lib)

task2: task2.c
	gcc -o  $@ task2.c -lm $(lib) 

task3: task3.cpp
	g++ -o task3_guided task3.cpp $(lib) -DTYPEGUIDED
	g++ -o task3_static task3.cpp $(lib) -DTYPESTATIC
	g++ -o task3_dynamic task3.cpp $(lib) -DTYPEDYNAMIC
