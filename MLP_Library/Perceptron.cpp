#include "Perceptron.h"
#include "RNG.h"



void Perceptron::SaveWeightsBias()
{
}

void Perceptron::LoadWeightsBias()
{
}

Perceptron::Perceptron(int nInputs)
{
	numInputs = nInputs;

	//TODO: Randomise bias
	bias = 0.0;
	output = 0.0;
	errorGradient = 0.0;
	N = 0.0;

	for(int i = 0; i < nInputs; ++i)
	{
		//TODO: randomise weights
		weights.push_back(0.0);
	}
}

Perceptron::~Perceptron()
{
}
