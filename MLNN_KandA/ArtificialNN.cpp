#include "ArtificialNN.h"

#include "RNG.h"
#include "ActivationFunctions.h"

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

	for (int n = 0; n < numberNeuronHiddenLayer; n++)
	{
		for (int l = 0; l < numberHiddenLayer; l++)
		{
			weights[l * numberNeuronHiddenLayer + n] = RNG::GetNumber();
		}
		//One bias per neuron
		biases[n] = RNG::GetNumber();
	}
}

ArtificialNN::ArtificialNN(int numberInput, int numberOutput, int numberHiddenLayer,
	std::vector<int> numberNeuronPerHiddenLayer, int OutputLearningRate, std::vector<double> learningRatePerHiddenLayer,
	std::vector<ACTIVATION_FUNCTION> af_PerHiddenLayer, ACTIVATION_FUNCTION af_OutputLayer)
{
	numInputs = numberInput;
	numOutputs = numberOutput;
	numNPerHidden = numberNeuronPerHiddenLayer;

	learningRateOutput = OutputLearningRate;
	learningRatePerHidden = learningRatePerHiddenLayer;

	activationFunctionHiddenLayer = af_PerHiddenLayer;
	activationFunctionOutputLayer = af_OutputLayer;
}

std::vector<double> ArtificialNN::Train(std::vector<double> inputValues, std::vector<double> desiredOutput)
{
	//Assuming all hidden layers have the same amount of neurons
	int neuronsPerHidden = numNPerHidden[0];
	//Assuming all hidden layers use the same activation functions
	ACTIVATION_FUNCTION hiddenActivation = activationFunctionHiddenLayer[0];

	std::vector<double> preActivation(numNPerHidden.size() * neuronsPerHidden);


	for (int i = 0; i < neuronsPerHidden; i++)
	{
		double tempSum = 0;
		for (int iV = 0; i < inputValues.size(); iV++)
		{
			tempSum += weights[iV + i * numInputs] * iV;
		}
		tempSum += biases[i];

		preActivation[i] = tempSum;
	}
	

	for (int l = 1; l < neuronsPerHidden; l++)
	{
		//For each neuron in the layer
		for (int n = 0; n < numNPerHidden[l]; n++)
		{
			double tempSum = 0;
			//For each input, which means each previous neuron
			for (int previousN = 0; previousN < numNPerHidden[l]; previousN++){
				
				tempSum += (weights[l * neuronsPerHidden + previousN] * MLNN_KandA::Math::ActivationFunction(hiddenActivation, preActivation[(l - 1) * neuronsPerHidden + previousN]));
				
			}
			tempSum += biases[l * neuronsPerHidden + n];

			preActivation[l * neuronsPerHidden + n] = tempSum;
		}
	}


	return std::vector<double>(numOutputs, 0 );
}


