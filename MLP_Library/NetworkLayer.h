#pragma once

#include <memory> 
#include <vector> 

#include "Perceptron.h" 

// Define the NetworkLayer class to represent a layer in a neural network
class NetworkLayer
{
public:
	int numNeurons; // Holds the number of neurons in this layer

	// Vector of shared_ptr to Perceptron objects. 
	std::vector<std::shared_ptr<Perceptron>> neurons;

	// Constructor that initializes a network layer with a specified number of neurons,
	// each neuron having a specified number of inputs
	NetworkLayer(int nNeurons, int numNeuronInputs);

	// Destructor
	~NetworkLayer();

	// Saves the weights and bias of the current layer to a file (not implemented here)
	void SaveWeightsBias();
	// Loads the weights and bias of the current layer from a file (not implemented here)
	void LoadWeightsBias();

};
