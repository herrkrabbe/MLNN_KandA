#include "ArtificialNN.h"

#include "RNG.h"
#include "ActivationFunctions.h"

using namespace	MLNN_KandA;

ArtificialNN::ArtificialNN(size_t numberInput, size_t numberOutput, size_t numberHiddenLayer, size_t numberNeuronHiddenLayer,
                           double OutputLearningRate, double learningRate, Math::eActivationFunction af_HiddenLayer, Math::eActivationFunction af_OutputLayer)
{
	numInputs = numberInput;
	numOutputs = numberOutput;
	numNPerHidden = std::vector<size_t>(numberHiddenLayer, numberNeuronHiddenLayer);

	learningRateOutput = OutputLearningRate;
	learningRatePerHidden = std::vector<double>(numberHiddenLayer, learningRate);

	activationFunctionHiddenLayer = std::vector<Math::eActivationFunction>(numberHiddenLayer, af_HiddenLayer);
	activationFunctionOutputLayer = af_OutputLayer;

	//setup hidden layer start indices
	for (size_t hLayer = 0; hLayer < numNPerHidden.size(); ++hLayer)
	{
		if (hLayer > 1)
		{
			weightHiddenLayerStartIndex.push_back(weightHiddenLayerStartIndex.at(hLayer - 1) + numNPerHidden.at(hLayer - 1) * numNPerHidden.at(hLayer - 2));
			biasHiddenLayerStartIndex.push_back(biasHiddenLayerStartIndex.at(hLayer - 1) + numNPerHidden.at(hLayer - 1));
		}
		else if (hLayer == 1)
		{
			weightHiddenLayerStartIndex.push_back(numberInput * numNPerHidden.at(0));
			biasHiddenLayerStartIndex.push_back(numNPerHidden.at(0));
		}
		else
		{
			weightHiddenLayerStartIndex.push_back(0);
			biasHiddenLayerStartIndex.push_back(0);
		}
	}

	//initializing output layer

		//OLD kittel code. replaced with new more readable approach
		// weightOutputStartIndex = numInputs * numberNeuronHiddenLayer + (numberHiddenLayer - 1) * numberNeuronHiddenLayer * numberNeuronHiddenLayer;
	weightOutputStartIndex =
		weightHiddenLayerStartIndex.back()
		+ weightHiddenLayerStartIndex.at(numNPerHidden.at(numNPerHidden.size() - 1))
		* weightHiddenLayerStartIndex.at(numNPerHidden.at(numNPerHidden.size() - 2));
	biasOutputStartIndex =
		biasHiddenLayerStartIndex.back()
		+ biasHiddenLayerStartIndex.at(numNPerHidden.at(numNPerHidden.size() - 1));

	size_t weightsSize = weightOutputStartIndex + numberOutput * weightHiddenLayerStartIndex.back();
	size_t biasesSize = biasOutputStartIndex + numberOutput;

	weights.resize(weightsSize);
	biases.resize(biasesSize);

	for (size_t i = 0; i < weightsSize; ++i)
	{
		weights.push_back(RNG::GetNumber());
	}
	for (size_t i = 0; i < biasesSize; ++i)
	{
		biases.push_back(RNG::GetNumber());
	}



	////Initializing first layer
	//for (int n = 0; n < numberNeuronHiddenLayer; n++)
	//{
	//	biases[n] = RNG::GetNumber();
	//	//one weight per input per first layer neuron
	//	for (int i = 0; i < numInputs; i++)
	//	{
	//		weights[n * numInputs + i] = RNG::GetNumber();
	//	}
	//}
	//

	////For each layer hidden
	//for (int l = 1; l < numberHiddenLayer; l++)
	//{
	//	//For each Neuron
	//	for (int n = 0; n < numberNeuronHiddenLayer; n++)
	//	{
	//		biases[numInputs + (l-1) * numberNeuronHiddenLayer + n] = RNG::GetNumber();

	//		//For each input to each neuron
	//		for (int i = 0; i < numberNeuronHiddenLayer; i++) 
	//		{
	//			weights[(numInputs * numberNeuronHiddenLayer) + (l-1) * numberNeuronHiddenLayer * numberNeuronHiddenLayer
	//				+ n * numberNeuronHiddenLayer + i] = RNG::GetNumber();
	//		}
	//		
	//	}
	//	//One bias per neuron
	//	
	//}

	//
	//

	////For each output
	//for (int o = 0; 0 < numOutputs; o++)
	//{
	//	biases[numInputs + (numberHiddenLayer - 1) * numberNeuronHiddenLayer + o] = RNG::GetNumber();

	//	//For each input to
	//	for (int n = 0; n < numberNeuronHiddenLayer; n++)
	//	{
	//		weights[weightOutputStartIndex + o * numberNeuronHiddenLayer + n];
	//	}
	//}

}

ArtificialNN::ArtificialNN(size_t numberInput, size_t numberOutput,
	std::vector<size_t> numberNeuronPerHiddenLayer, double OutputLearningRate, std::vector<double> learningRatePerHiddenLayer,
	std::vector<Math::eActivationFunction> af_PerHiddenLayer, Math::eActivationFunction af_OutputLayer)
{
	numInputs = numberInput;
	numOutputs = numberOutput;
	numNPerHidden = numberNeuronPerHiddenLayer;

	learningRateOutput = OutputLearningRate;
	learningRatePerHidden = learningRatePerHiddenLayer;

	activationFunctionHiddenLayer = af_PerHiddenLayer;
	activationFunctionOutputLayer = af_OutputLayer;

	//setup hidden layer start indices
	for (size_t hLayer = 0; hLayer < numNPerHidden.size(); ++hLayer)
	{
		if (hLayer > 1)
		{
			weightHiddenLayerStartIndex.push_back(weightHiddenLayerStartIndex.at(hLayer - 1) + numNPerHidden.at(hLayer - 1) * numNPerHidden.at(hLayer - 2));
			biasHiddenLayerStartIndex.push_back(biasHiddenLayerStartIndex.at(hLayer - 1) + numNPerHidden.at(hLayer - 1));
		}
		else if (hLayer == 1)
		{
			weightHiddenLayerStartIndex.push_back(numberInput * numNPerHidden.at(0));
			biasHiddenLayerStartIndex.push_back(numNPerHidden.at(0));
		}
		else
		{
			weightHiddenLayerStartIndex.push_back(0);
			biasHiddenLayerStartIndex.push_back(0);
		}
	}
	weightOutputStartIndex =
		weightHiddenLayerStartIndex.back()
		+ weightHiddenLayerStartIndex.at(numNPerHidden.at(numNPerHidden.size() - 1))
		* weightHiddenLayerStartIndex.at(numNPerHidden.at(numNPerHidden.size() - 2));
	biasOutputStartIndex =
		biasHiddenLayerStartIndex.back()
		+ biasHiddenLayerStartIndex.at(numNPerHidden.at(numNPerHidden.size() - 1));

	size_t weightsSize = weightOutputStartIndex + numberOutput * weightHiddenLayerStartIndex.back();
	size_t biasesSize = biasOutputStartIndex + numberOutput;

	weights.resize(weightsSize);
	biases.resize(biasesSize);

	for(size_t i=0; i < weightsSize; ++i)
	{
		weights.push_back(RNG::GetNumber());
	}
	for (size_t i = 0; i < biasesSize; ++i)
	{
		biases.push_back(RNG::GetNumber());
	}
}

std::vector<double> ArtificialNN::Train(std::vector<double> inputValues, std::vector<double> desiredOutput)
{
	//Assuming all hidden layers have the same amount of neurons
	size_t neuronsPerHidden = numNPerHidden[0];
	//Assuming all hidden layers use the same activation functions
	Math::eActivationFunction hiddenActivation = activationFunctionHiddenLayer[0];

	std::vector<double> preActivation(numNPerHidden.size() * neuronsPerHidden);

	for (size_t i = 0; i < neuronsPerHidden; i++)
	{
		double tempSum = 0;
		for (size_t iV = 0; i < inputValues.size(); iV++)
		{
			tempSum += weights[iV + i * numInputs] * iV;
		}
		tempSum += biases[i];

		preActivation[i] = tempSum;
	}

	//	l * neuronsPerHidden = firstLayerSize + (l-1) * neuronsPerHidden

	for (size_t l = 1; l < numNPerHidden.size(); l++)
	{
		//For each neuron in the layer
		for (size_t n = 0; n < neuronsPerHidden; n++)
		{
			double tempSum = 0;
			//For each input, which means each previous neuron
			for (size_t previousN = 0; previousN < neuronsPerHidden; previousN++){
				
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

	for(size_t o = 0; o < numOutputs; o++)
	{
		double tempSum = 0;
		for (size_t n = 0; n < neuronsPerHidden; n++)
		{
			tempSum += weights[weightOutputStartIndex + o * neuronsPerHidden + n]
				* Math::ActivationFunction(hiddenActivation, preActivation[numInputs + (numNPerHidden.size() - 2) * neuronsPerHidden] + n);
		}

		tempSum += biases[numInputs + (numNPerHidden.size() - 2) * neuronsPerHidden + o];

		preActivation[numInputs + (numNPerHidden.size() - 2) * neuronsPerHidden + o] = tempSum;
	}
	//Output achieved

	//Comparing output to desired output
	for (size_t o = 0; o < numOutputs; o++)
	{
		double error = desiredOutput[o] - Math::ActivationFunction(activationFunctionOutputLayer, 
			preActivation[numInputs + (numNPerHidden.size() - 2) * neuronsPerHidden + o]);
	}




	return std::vector<double>(numOutputs, 0 );
}


