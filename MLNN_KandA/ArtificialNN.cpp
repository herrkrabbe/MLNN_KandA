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
	for (int n = 0; n < numberNeuronHiddenLayer; n++)
	{
		biases[n] = RNG::GetNumber();
		for (int i = 0; i < numInputs; i++)
		{
			weights[n * numInputs + i] = RNG::GetNumber();
		}
	}
	

	//For each layer
	for (int l = 1; l < numberHiddenLayer; l++)
	{
		//For each Neuron
		for (int n = 0; n < numberNeuronHiddenLayer; n++)
		{
			biases[numInputs + (l-1) * numberNeuronHiddenLayer + n] = RNG::GetNumber();

			//For each input to each neuron
			for (int i = 0; i < numberNeuronHiddenLayer; i++) 
			{
				weights[(numInputs * numberNeuronHiddenLayer) + (l-1) * numberNeuronHiddenLayer * numberNeuronHiddenLayer
					+ n * numberNeuronHiddenLayer + i] = RNG::GetNumber();
			}
			
		}
		//One bias per neuron
		
	}

	//initializing output layer

	startPositionOutput = numInputs * numberNeuronHiddenLayer + (numberHiddenLayer - 1) * numberNeuronHiddenLayer * numberNeuronHiddenLayer;

	//For each output
	for (int o = 0; 0 < numOutputs; o++)
	{
		biases[numInputs + (numberHiddenLayer - 1) * numberNeuronHiddenLayer + o] = RNG::GetNumber();

		//For each input to
		for (int n = 0; n < numberNeuronHiddenLayer; n++)
		{
			weights[startPositionOutput + o * numberNeuronHiddenLayer + n];
		}
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
					weights[(numInputs * neuronsPerHidden) + (l-1) * neuronsPerHidden * neuronsPerHidden 
					+ n * neuronsPerHidden + previousN] 
					* Math::ActivationFunction(hiddenActivation, preActivation[numInputs + (l - 1) * neuronsPerHidden + previousN]);
				
			}
			tempSum += biases[numInputs + (l-1) * neuronsPerHidden + n];

			preActivation[l * neuronsPerHidden + n] = tempSum;
		}
	}
	//Finished calculating forward

	for(int o = 0; o < numOutputs; o++)
	{
		double tempSum = 0;
		for (int n = 0; n < neuronsPerHidden; n++)
		{
			tempSum += weights[startPositionOutput + o * neuronsPerHidden + n]
				* Math::ActivationFunction(hiddenActivation, preActivation[numInputs + (numNPerHidden.size() - 2) * neuronsPerHidden] + n);
		}

		tempSum += biases[numInputs + (numNPerHidden.size() - 2) * neuronsPerHidden + o];

		preActivation[numInputs + (numNPerHidden.size() - 2) * neuronsPerHidden + o] = tempSum;
	}
	//Output achieved

	//Comparing output to desired output
	for (int o = 0; o < numOutputs; o++)
	{
		double error = desiredOutput[o] - Math::ActivationFunction(activationFunctionOutputLayer, 
			preActivation[numInputs + (numNPerHidden.size() - 2) * neuronsPerHidden + o]);
	}




	return std::vector<double>(numOutputs, 0 );
}


