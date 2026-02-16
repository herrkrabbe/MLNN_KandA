#pragma once

#include <memory> 
#include <vector> 

#include "Neuron_KTAN.h" 

using namespace std; 

// Define the NetworkLayer_KTAN class to represent a layer in a neural network
class NetworkLayer_KTAN
{
public:
    int numNeurons; // Holds the number of neurons in this layer

    // Vector of shared_ptr to Neuron_KTAN objects. 
    vector<shared_ptr<Neuron_KTAN>> neurons;

    // Constructor that initializes a network layer with a specified number of neurons,
    // each neuron having a specified number of inputs
    NetworkLayer_KTAN(int nNeurons, int numNeuronInputs);
    
    // Destructor
    ~NetworkLayer_KTAN();

    // Saves the weights and bias of the current layer to a file (not implemented here)
    void SaveWeightsBias();
    // Loads the weights and bias of the current layer from a file (not implemented here)
    void LoadWeightsBias();

};
