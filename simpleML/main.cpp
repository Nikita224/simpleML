#include <fstream>
#include <thread>
#include <random>
#include <time.h>
#include <Windows.h>
#include <iostream>

using namespace std;

struct neuron {
	double value;
	double error;
	void act() {
		value = (1 / (1 + pow(2.71828, -value)));
	}
};

class network {
public:
	int layers;
	neuron** neurons;
	double*** weights;

};