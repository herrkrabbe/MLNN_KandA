#include "ArtificialNN.h"

#include "RNG.h"
#include "ActivationFunctions.h"

using namespace	MLNN_KandA;

ArtificialNN::ArtificialNN(int numberInput, int numberOutput, int numberHiddenLayer, int numberNeuronHiddenLayer,
                           int OutputLearningRate, double learningRate, Math::eActivationFunction af_HiddenLayer, Math::eActivationFunction af_OutputLayer)
{
	numInputs = numberInput;
	numOutputs = numberOutput;
	numNPerHidden = std::vector<int>(numberHiddenLayer, numberNeuronHiddenLayer);

	learningRateOutput = OutputLearningRate;
	learningRatePerHidden = std::vector<double>(numberHiddenLayer, learningRate);

	activationFunctionHiddenLayer = std::vector<Math::eActivationFunction>(numberHiddenLayer, af_HiddenLayer);
	activationFunctionOutputLayer = af_OutputLayer;

	//Initializing first layer
	int firstLayerSize = 0;
	for (int n = 0; n < numberNeuronHiddenLayer; n++)
	{
		biases[n] = RNG::GetNumber();
		for (int i = 0; i < numInputs; i++)
		{
			weights[n * numInputs + i] = RNG::GetNumber();
			firstLayerSize++;
		}
	}
	

	//For each layer
	for (int l = 1; l < numberHiddenLayer; l++)
	{
		//For each Neuron
		for (int n = 0; n < numberNeuronHiddenLayer; n++)
		{
			biases[firstLayerSize + (l-1) * numberNeuronHiddenLayer + n] = RNG::GetNumber();

			//For each input to each neuron
			for (int i = 0; i < numberNeuronHiddenLayer; i++) 
			{
				weights[firstLayerSize + (l-1) * numberNeuronHiddenLayer + n] = RNG::GetNumber();
			}
			
		}
		//One bias per neuron
		
	}
}

ArtificialNN::ArtificialNN(int numberInput, int numberOutput, int numberHiddenLayer,
	std::vector<int> numberNeuronPerHiddenLayer, int OutputLearningRate, std::vector<double> learningRatePerHiddenLayer,
	std::vector<Math::eActivationFunction> af_PerHiddenLayer, Math::eActivationFunction af_OutputLayer)
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
	Math::eActivationFunction hiddenActivation = activationFunctionHiddenLayer[0];

	std::vector<double> preActivation(numNPerHidden.size() * neuronsPerHidden);

	int firstLayerSize = 0;
	for (int i = 0; i < neuronsPerHidden; i++)
	{
		double tempSum = 0;
		for (int iV = 0; i < inputValues.size(); iV++)
		{
			tempSum += weights[iV + i * numInputs] * iV;
			firstLayerSize++;
		}
		tempSum += biases[i];

		preActivation[i] = tempSum;
	}

	//	l * neuronsPerHidden = firstLayerSize + (l-1) * neuronsPerHidden

	for (int l = 1; l < numNPerHidden.size(); l++)
	{
		//For each neuron in the layer
		for (int n = 0; n < neuronsPerHidden; n++)
		{
			double tempSum = 0;
			//For each input, which means each previous neuron
			for (int previousN = 0; previousN < neuronsPerHidden; previousN++){
				
				tempSum += 
					weights[firstLayerSize + (l-1) * neuronsPerHidden * neuronsPerHidden + n * neuronsPerHidden + previousN] 
					* Math::ActivationFunction(hiddenActivation, preActivation[(l - 1) * neuronsPerHidden + previousN]);
				
			}
			tempSum += biases[l * neuronsPerHidden + n];

			preActivation[l * neuronsPerHidden + n] = tempSum;
		}
	}


	return std::vector<double>(numOutputs, 0 );
}


