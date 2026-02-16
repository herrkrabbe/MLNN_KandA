#pragma once

#include <vector> 

// Define a class representing a single neuron in a neural network
class Perceptron
{
public:
	int numInputs;         // Number of inputs the neuron receives
	double bias;           // Bias value for the neuron, used in its activation function
	double output;         // The output value of the neuron after processing inputs
	double errorGradient;  // The gradient of the error for this neuron, used during backpropagation
	double N;              // The value before activation function is applied (also known as the net input)

	std::vector<double> weights; // Dynamic array of weights for each input
	std::vector<double> inputs;  // Dynamic array of inputs received by the neuron

	// Saves the weights and bias of the current neuron to a file (not implemented here)
	void SaveWeightsBias();
	// Loads the weights and bias of the current neuron from a file (not implemented here)
	void LoadWeightsBias();


	// Constructor that initializes a neuron with a specific number of inputs
	Perceptron(int nInputs);

	// Destructor
	~Perceptron();
};