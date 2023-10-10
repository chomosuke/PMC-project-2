solution: solution.cc
	mpicxx -O3 -fopenmp solution.cc -o solution -std=c++20

example: example.cc
	mpicxx -O3 -fopenmp example.cc -o example -std=c++20

debug: solution.cc
	mpicxx -O0 -fopenmp solution.cc -o debug -std=c++20 -g

clean:
	rm solution PQ-Dijkstra -f
