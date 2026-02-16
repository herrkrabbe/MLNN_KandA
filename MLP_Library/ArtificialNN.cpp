#include "ArtificialNN.h"

ArtificialNN::ArtificialNN(int numberInput, int numberOutput, int numberHiddenLayer, int numberNeuronHiddenLayer,
	int OutputLearningRate, double learningRate, ACTIVATION_FUNCTION af_HiddenLayer, ACTIVATION_FUNCTION af_OutputLayer)
{
	numInputs = numberInput;
	numOutputs = numberOutput;
	numNPerHidden = std::vector<int>(numberHiddenLayer, numberNeuronHiddenLayer);

	learningRateOutput = OutputLearningRate;
	learningRatePerHidden = std::vector<double>(numberHiddenLayer, learningRate);

	activationFunctionHiddenLayer = std::vector<ACTIVATION_FUNCTION>(numberHiddenLayer, af_HiddenLayer);
	activationFunctionOutputLayer = af_OutputLayer;
}

ArtificialNN::ArtificialNN(int numberInput, int numberOutput, int numberHiddenLayer,
	std::vector<int> numberNeuronPerHiddenLayer, int OutputLearningRate, std::vector<double> learningRatePerHiddenLayer,
	std::vector<ACTIVATION_FUNCTION> af_PerHiddenLayer, ACTIVATION_FUNCTION af_OutputLayer)
{
	numInputs = numberInput;
	numOutputs = numberOutput;
	numNPerHidden = numberNeuronPerHiddenLayer;

	learningRateOutput = OutputLearningRate;
	learningRatePerHidden = learningRatePerHidden;

	activationFunctionHiddenLayer = af_PerHiddenLayer;
	activationFunctionOutputLayer = af_OutputLayer;
}
