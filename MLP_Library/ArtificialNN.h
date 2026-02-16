#pragma once

#include <string>
#include <vector>

/**
 *
 */

class ArtificialNN
{
public:
	enum ACTIVATION_FUNCTION
	{
		BINARY_STEP
		, SIGMOID
		, TANH
		, RELU
		, LEAKY_RELU
	};

	int numInputs;
	int numOutputs;
	//int numHidden;		Instead use		=>	 numNPerHidden.size()
	std::vector<int> numNPerHidden;

	std::vector<double> learningRate;
	std::vector<double> learningRatePerHidden;

	//Identical layer amount variant
	ArtificialNN(int numberInput, int numberOutput,
		int numberHiddenLayer, int numberNeuronPerHideenLayer, double learningRate,
		ACTIVATION_FUNCTION af_HiddenLayer, ACTIVATION_FUNCTION af_OutputLayer);

	//Varied Neurons per hidden layer variant
	ArtificialNN(int numberInput, int numberOutput,
		int numberHiddenLayer, std::vector<int> numberNeuronPerHiddenLayer, int OutputLearningRate, std::vector<double> learningRatePerHiddenLayer,
		ACTIVATION_FUNCTION af_HiddenLayer, ACTIVATION_FUNCTION af_OutputLayer);

	~ArtificialNN();

	//Train method is to compute the output + update weight
	std::vector<double> Train(std::vector<double> inputValues, std::vector<double> desiredOutput);

	//CalcOutput method is only to compute the output (without update weight for training)
	std::vector<double> CalcOutput(std::vector<double> inputValues);

	std::string PrintWeightsBias();

	// Saves the weights and bias of the current network to a file (not implemented here)
	void SaveWeightsBias();
	// Loads the weights and bias of the current network from a file (not implemented here)
	void LoadWeightsBias();

private:

	ACTIVATION_FUNCTION activationFunctionHiddenLayer;
	ACTIVATION_FUNCTION activationFunctionOutputLayer;

	// perform backpropagation to update weights
	void UpdateWeights(std::vector<double> outputs, std::vector<double> desiredOutput);


	double ActivationFunctionH(double value);	//hidden layer
	double ActivationFunctionO(double value);	//output layer

	double Step(double value);

	double TanH(double value);

	double Sigmoid(double value);

	double ReLU(double value);

	double LeakyReLU(double value);

	double Derivated_Activation_Function(ACTIVATION_FUNCTION af, double value);

};
