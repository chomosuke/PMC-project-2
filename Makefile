solution: solution.cc
	mpicxx -O3 solution.cc -o solution -std=c++20

debug: solution.cc
	mpicxx -O0 solution.cc -o debug -std=c++20 -g

clean:
	rm solution debug -f
