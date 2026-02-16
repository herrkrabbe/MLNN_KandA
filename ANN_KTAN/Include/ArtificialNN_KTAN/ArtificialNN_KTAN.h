// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include <string>
#include <vector>

#include "NetworkLayer_KTAN.h"
/**
 * 
 */
using namespace::std;

class ArtificialNN_KTAN
{
public:
	enum ACTIVATION_FUNCTION
	{
		BINARY_STEP, SIGMOID, TANH, RELU, LEAKY_RELU
	};

	int numInputs;
	int numOutputs;
	int numHidden;
	int numNPerHidden;
	double learningRate;
	
	vector<shared_ptr<NetworkLayer_KTAN>> layers;

	ArtificialNN_KTAN(	int numberInput, int numberOutput, 
						int numberHiddenLayer, int numberNeuronPerHideenLayer, double learningRate,
						ACTIVATION_FUNCTION af_HiddenLayer, ACTIVATION_FUNCTION af_OutputLayer);
	~ArtificialNN_KTAN();

	//Train method is to compute the output + update weight
	vector<double> Train(vector<double> inputValues, vector<double> desiredOutput);

	//CalcOutput method is only to compute the output (without update weight for training)
	vector<double> CalcOutput(vector<double> inputValues);

	string PrintWeightsBias();

	// Saves the weights and bias of the current network to a file (not implemented here)
	void SaveWeightsBias();
	// Loads the weights and bias of the current network from a file (not implemented here)
	void LoadWeightsBias();

private:

	ACTIVATION_FUNCTION activationFunctionHiddenLayer;
	ACTIVATION_FUNCTION activationFunctionOutputLayer;

	// perform backpropagation to update weights
	void UpdateWeights(vector<double> outputs, vector<double> desiredOutput);


	double ActivationFunctionH(double value);	//hidden layer
	double ActivationFunctionO(double value);	//output layer

	double Step(double value);

	double TanH(double value);

	double Sigmoid(double value);

	double ReLU(double value);

	double LeakyReLU(double value);

	double Derivated_Activation_Function(ACTIVATION_FUNCTION af, double value);

};
