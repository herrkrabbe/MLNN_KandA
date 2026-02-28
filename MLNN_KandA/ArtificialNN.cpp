#include "ArtificialNN.h"

#include "RNG.h"
#include "ActivationFunctions.h"
#include <iostream>
#include <string>
#include <sstream>

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
	{
		size_t backIndex = weightHiddenLayerStartIndex.back();

		size_t foo11 = numNPerHidden.size() - 1;
		size_t foo12 = numNPerHidden.at(foo11);

		size_t foo21 = numNPerHidden.size() - 2;
		size_t foo22 = numNPerHidden.at(foo21);

		size_t sizeOfLastLayer = foo12 * foo22;

		weightOutputStartIndex =
			backIndex
			+ sizeOfLastLayer;
	}
	{
		biasOutputStartIndex =
			biasHiddenLayerStartIndex.back()
			+ numNPerHidden.at(numNPerHidden.size() - 1);
	}
	

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
	{
		size_t backIndex = weightHiddenLayerStartIndex.back();

		size_t foo11 = numNPerHidden.size() - 1;
		size_t foo12 = numNPerHidden.at(foo11);

		size_t foo21 = numNPerHidden.size() - 2;
		size_t foo22 = numNPerHidden.at(foo21);

		size_t sizeOfLastLayer = foo12 * foo22;

		weightOutputStartIndex =
			backIndex
			+ sizeOfLastLayer;
	}
	{
		biasOutputStartIndex =
			biasHiddenLayerStartIndex.back()
			+ numNPerHidden.at(numNPerHidden.size() - 1);
	}

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

//std::vector<double> ArtificialNN::Train(std::vector<double> inputValues, std::vector<double> desiredOutput)
//{
//	//Assuming all hidden layers have the same amount of neurons
//	size_t neuronsPerHidden = numNPerHidden[0];
//	//Assuming all hidden layers use the same activation functions
//	Math::eActivationFunction hiddenActivation = activationFunctionHiddenLayer[0];
//
//	std::vector<double> preActivation(numNPerHidden.size() * neuronsPerHidden);
//
//	for (size_t i = 0; i < neuronsPerHidden; i++)
//	{
//		double tempSum = 0;
//		for (size_t iV = 0; i < inputValues.size(); iV++)
//		{
//			tempSum += weights[iV + i * numInputs] * iV;
//		}
//		tempSum += biases[i];
//
//		preActivation[i] = tempSum;
//	}
//
//	//	l * neuronsPerHidden = firstLayerSize + (l-1) * neuronsPerHidden
//
//	for (size_t l = 1; l < numNPerHidden.size(); l++)
//	{
//		//For each neuron in the layer
//		for (size_t n = 0; n < neuronsPerHidden; n++)
//		{
//			double tempSum = 0;
//			//For each input, which means each previous neuron
//			for (size_t previousN = 0; previousN < neuronsPerHidden; previousN++){
//				
//				tempSum +=
//					weights[(numInputs * neuronsPerHidden) + (l-1) * neuronsPerHidden * neuronsPerHidden 
//					+ n * neuronsPerHidden + previousN] 
//					* Math::ActivationFunction(hiddenActivation, preActivation[numInputs + (l - 1) * neuronsPerHidden + previousN]);
//				
//			}
//			tempSum += biases[numInputs + (l-1) * neuronsPerHidden + n];
//
//			preActivation[l * neuronsPerHidden + n] = tempSum;
//		}
//	}
//	//Finished calculating forward
//
//	for(size_t o = 0; o < numOutputs; o++)
//	{
//		double tempSum = 0;
//		for (size_t n = 0; n < neuronsPerHidden; n++)
//		{
//			tempSum += weights[weightOutputStartIndex + o * neuronsPerHidden + n]
//				* Math::ActivationFunction(hiddenActivation, preActivation[numInputs + (numNPerHidden.size() - 2) * neuronsPerHidden] + n);
//		}
//
//		tempSum += biases[numInputs + (numNPerHidden.size() - 2) * neuronsPerHidden + o];
//
//		preActivation[numInputs + (numNPerHidden.size() - 2) * neuronsPerHidden + o] = tempSum;
//	}
//	//Output achieved
//
//	//Comparing output to desired output
//	for (size_t o = 0; o < numOutputs; o++)
//	{
//		double error = desiredOutput[o] - Math::ActivationFunction(activationFunctionOutputLayer, 
//			preActivation[numInputs + (numNPerHidden.size() - 2) * neuronsPerHidden + o]);
//	}
//
//
//
//
//	return std::vector<double>(numOutputs, 0 );
//}

std::vector<double> ArtificialNN::Train(std::vector<double> inputValues, std::vector<double> desiredOutput)
{
	//Assuming all hidden layers have the same amount of neurons
	size_t neuronsPerHidden = numNPerHidden[0];
	//Assuming all hidden layers use the same activation functions
	Math::eActivationFunction hiddenActivation = activationFunctionHiddenLayer[0];

	std::vector<double> preActivation(numNPerHidden.size() * neuronsPerHidden+numOutputs);

	// first input layer preactivation calculation
	for (size_t i = 0; i < numNPerHidden[0]; i++)
	{
		double tempSum = 0;
		for (size_t iV = 0; iV < inputValues.size(); iV++)
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
		for(size_t n = 0; n < numNPerHidden[l]; ++n)
		{
			double tempSum = 0;
			//For each input, which means each previous neuron
			size_t numNeuronsPrevious = numNPerHidden[l - 1];
			for (size_t previousN = 0; previousN < numNeuronsPrevious; previousN++) {
				size_t weightOfPreviousN = GetWeightHiddenLayerStartIndex(l) + n * numNeuronsPrevious + previousN;
				size_t previousNeuronPreActivationIndex = GetBiasHiddenLayerStartIndex(l-1)+previousN;

				tempSum +=
					weights[weightOfPreviousN]
					* Math::ActivationFunction(hiddenActivation, preActivation[previousNeuronPreActivationIndex]);

			}
			tempSum += biases[GetBiasHiddenLayerStartIndex(l)];

			size_t preActivationIndex = GetBiasHiddenLayerStartIndex(l)+n;

			preActivation[preActivationIndex] = tempSum;
		}
	}
	//Finished calculating forward

	for (size_t o = 0; o < numOutputs; o++)
	{
		double tempSum = 0;
		for (size_t n = 0; n < numNPerHidden.back(); n++)
		{
			size_t weightOfPreviousN = GetWeightOutputStartIndex() + o * numNPerHidden.back() + n;
			size_t lastLayerNumber = numNPerHidden.size()-1;
			size_t previousNeuronPreActivationIndex = GetBiasHiddenLayerStartIndex(lastLayerNumber) + n;



			tempSum += weights[weightOfPreviousN]
				* Math::ActivationFunction(activationFunctionOutputLayer, preActivation[previousNeuronPreActivationIndex]);
		}

		tempSum += biases[GetBiasOutputStartIndex() + o];

		preActivation[GetBiasOutputStartIndex() + o] = tempSum;
	}
	//Output achieved


	std::vector<double> errorGradient(biases.size(), 0.0);
	//calculate error gradient for output
	for (size_t o = 0; o < numOutputs; o++)
	{
		size_t outputIndex = GetBiasOutputStartIndex()+o;
		double calculatedOutput = ActivationFunction(activationFunctionOutputLayer, preActivation[outputIndex]);
		double errorDifference = desiredOutput[o] - calculatedOutput;
		double outputErrorGadient = errorDifference * DerivedFunction(activationFunctionOutputLayer, calculatedOutput);
		errorGradient[outputIndex] = outputErrorGadient;
	}

	//calculate error gradient for hidden layers
	{
		size_t l = numHidden - 1;
		size_t numberOfNeurons = numNPerHidden[l];
		for(size_t n = 0; n < numberOfNeurons; ++n)
		{
			size_t neuronIndex = GetBiasHiddenLayerStartIndex(l)+n;
			for(size_t o = 0; o < numOutputs; ++o)
			{
				size_t outputIndex = GetBiasOutputStartIndex() + o;
				double outputErrorGradient = errorGradient[outputIndex];
				size_t weightIndex = GetWeightOutputStartIndex() + o * numNPerHidden.back() + n;
				double weight = weights[weightIndex];
				double errorGradientPart = weight * outputErrorGradient;
				errorGradient[neuronIndex] += errorGradientPart;
			}
			errorGradient[neuronIndex] *= ActivateThenDerive(hiddenActivation, preActivation[neuronIndex]);
		}

	}

	for (size_t foo = 0; foo < numHidden - 2; ++foo)
	{
		size_t l = numHidden - 2 - foo;
		size_t nextLayer = l+1;
		size_t numberOfNeurons = numNPerHidden[l];
		for (size_t n = 0; n < numberOfNeurons; ++n)
		{
			size_t neuronIndex = GetBiasHiddenLayerStartIndex(l) + n;
			for (size_t m = 0; m < numOutputs; ++m)
			{
				size_t nextLayerIndex = GetBiasHiddenLayerStartIndex(nextLayer) + m;
				double nextLayerErrorGradient = errorGradient[nextLayerIndex];
				size_t weightIndex = GetWeightHiddenLayerStartIndex(nextLayer) + m * numNPerHidden[l] + n;
				//size_t weightIndex = GetWeightHiddenLayerStartIndex(l+1) + m * numNPerHidden.back() + n;
				double weight = weights[weightIndex];
				double errorGradientPart = weight * nextLayerErrorGradient;
				errorGradient[neuronIndex] += errorGradientPart;
			}
			errorGradient[neuronIndex] *= ActivateThenDerive(hiddenActivation, preActivation[neuronIndex]);
		}
	}

	//Updating Weights and biases
	//First layer
	for (size_t n = 0; n < numHidden; n++)
	{
		for (size_t iV = 0; iV < inputValues.size(); iV++)
		{
			weights[iV + n * numInputs] = (	weights[iV + n * numInputs] + learningRatePerHidden[n] * errorGradient[n] * inputValues[iV]);
		}
		biases[n] = biases[n] + learningRatePerHidden[n] * errorGradient[n];
	}

	//Hidden Layer
	for (size_t l = 1; l < numNPerHidden.size(); l++)
	{
		size_t numNeuronsPrevious = numNPerHidden[l - 1];
		//For each neuron in the layer
		for (size_t n = 0; n < numNPerHidden[l]; ++n)
		{
			double tempSum = 0;
			//For each input, which means each previous neuron
			
			for (size_t previousN = 0; previousN < numNeuronsPrevious; previousN++) {
				//Jeg forstår ikke funksjonene dine, så jeg tror de er bugget/ikke komplett, bruker inntil videre hard logikk
				//size_t weightOfPreviousN = GetWeightHiddenLayerStartIndex(l) + n * numNeuronsPrevious + previousN;
				//size_t previousNeuronPreActivationIndex = GetBiasHiddenLayerStartIndex(l - 1) + previousN;

				weights[(numInputs * neuronsPerHidden) + (l - 1) * neuronsPerHidden * neuronsPerHidden+ n * neuronsPerHidden + previousN]
					= weights[(numInputs * neuronsPerHidden) + (l - 1) * neuronsPerHidden * neuronsPerHidden + n * neuronsPerHidden + previousN]
					+ learningRatePerHidden[numInputs + (l-1) * neuronsPerHidden + n]
					* errorGradient[l * neuronsPerHidden + n]
					* Math::ActivationFunction(activationFunctionHiddenLayer[0], preActivation[l - 1 * neuronsPerHidden + previousN]);
			}
			biases[numInputs + (l - 1) * neuronsPerHidden + n]
				= biases[numInputs + (l - 1) * neuronsPerHidden + n]
				+ learningRatePerHidden[l * neuronsPerHidden + n] * errorGradient[l * neuronsPerHidden + n];
		}
	}

	//Output Layer
	for (size_t o = 0; o < numOutputs; o++)
	{
		
		for (size_t n = 0; n < neuronsPerHidden; n++)
		{
			weights[weightOutputStartIndex + o * neuronsPerHidden + n]
				= weights[weightOutputStartIndex + o * neuronsPerHidden + n]
				+ learningRateOutput
				* errorGradient[numNPerHidden.size() * neuronsPerHidden + o]
				* Math::ActivationFunction(activationFunctionOutputLayer, preActivation[(numNPerHidden.size() - 1) * neuronsPerHidden + n]);
		}
		biases[numInputs + (numNPerHidden.size() - 2) * neuronsPerHidden + o]
			= biases[numInputs + (numNPerHidden.size() - 2) * neuronsPerHidden + o]
			+ learningRateOutput * errorGradient[numNPerHidden.size() * neuronsPerHidden + o];
		
		
		
		preActivation[numInputs + (numNPerHidden.size() - 2) * neuronsPerHidden + o];
	}


	return std::vector<double>(numOutputs, 0);
}

void MLNN_KandA::ArtificialNN::PrintLayerIndices()
{
	std::stringstream ss;
	for(int i = 0; i < weightHiddenLayerStartIndex.size(); ++i)
	{
		ss << "Layer: " << i << ", Weight start index: " << GetWeightHiddenLayerStartIndex(i) << ", Bias start index: " << GetBiasHiddenLayerStartIndex(i) << std::endl;
	}
	ss << "Output weight index: " << GetWeightOutputStartIndex() << ", Output bias index: " << GetBiasOutputStartIndex() << std::endl;
	std::cout << ss.str();
}


