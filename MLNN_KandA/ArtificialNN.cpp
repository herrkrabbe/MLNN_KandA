#include "ArtificialNN.h"

#include "RNG.h"
#include "ActivationFunctions.h"
#include <iostream>
#include <string>
#include <sstream>

using namespace	MLNN_KandA;

ArtificialNN::ArtificialNN(size_t numberInput, size_t numberOutput, 
	size_t numberHiddenLayer, size_t numberNeuronHiddenLayer,
    double OutputLearningRate, double learningRate, 
	Math::eActivationFunction af_HiddenLayer, 
	Math::eActivationFunction af_OutputLayer)
	:
	// Set constant values
	numInputs(numberInput)
	, numOutputs(numberOutput)
	, learningRateOutput(OutputLearningRate)
	, learningRatePerHidden(numberHiddenLayer, learningRate)
	, numNPerHidden(numberHiddenLayer, numberNeuronHiddenLayer)
	, activationFunctionHiddenLayer(numberHiddenLayer, af_HiddenLayer)
	, activationFunctionOutputLayer(af_OutputLayer)
{
	Init();
}

ArtificialNN::ArtificialNN(size_t numberInput, size_t numberOutput,
	std::vector<size_t> numberNeuronPerHiddenLayer, double OutputLearningRate, 
	std::vector<double> learningRatePerHiddenLayer,
	std::vector<Math::eActivationFunction> af_PerHiddenLayer, Math::eActivationFunction af_OutputLayer)
	:
	// Set constant values
	numInputs(numberInput)
	, numOutputs(numberOutput)
	, learningRateOutput(OutputLearningRate)
	, learningRatePerHidden(learningRatePerHiddenLayer)
	, numNPerHidden(numberNeuronPerHiddenLayer)
	, activationFunctionHiddenLayer(af_PerHiddenLayer)
	, activationFunctionOutputLayer(af_OutputLayer)
{
	Init();
}

void ArtificialNN::Init()
{
	//setup hidden layer start indices
	{
		weightLayerSize.resize(numHidden + 1); // resize to 1 index per layer + 1 for output

		weightHiddenLayerStartIndex.push_back(0);
		biasHiddenLayerStartIndex.push_back(0);
		weightLayerSize[0] = numInputs * numNPerHidden[0];
		for (size_t hLayer = 1; hLayer < numNPerHidden.size(); ++hLayer)
		{
			size_t prevIndex = hLayer - 1;
			weightHiddenLayerStartIndex.push_back(weightHiddenLayerStartIndex[prevIndex] + GetWeightLayerSize(prevIndex));
			biasHiddenLayerStartIndex.push_back(biasHiddenLayerStartIndex[prevIndex] + GetBiasLayerSize(prevIndex));
			weightLayerSize[hLayer] = numNPerHidden[prevIndex] * numNPerHidden[hLayer];
		}
	}


	//initializing output layer
	{
		size_t backIndex = weightHiddenLayerStartIndex.back();
		size_t sizeOfLastLayer = GetWeightLayerSize(numHidden-1);

		weightOutputStartIndex =
			backIndex
			+ sizeOfLastLayer;

		weightLayerSize.back() = numNPerHidden.back() * numOutputs;
	}
	{
		biasOutputStartIndex =
			biasHiddenLayerStartIndex.back()
			+ numNPerHidden.at(numHidden - 1);
	}

	//randomise initial weights and biases
	{
		size_t weightsSize = weightOutputStartIndex + numOutputs * numNPerHidden.back();
		size_t biasesSize = biasOutputStartIndex + numOutputs;

		weights.resize(weightsSize);
		biases.resize(biasesSize);
		preActivation.resize(biasesSize, 0.0);

		for (size_t i = 0; i < weightsSize; ++i)
		{
			weights[i] = RNG::GetNumber();
		}
		for (size_t i = 0; i < biasesSize; ++i)
		{
			biases[i] = RNG::GetNumber();
		}
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

std::vector<double> ArtificialNN::Train(std::vector<double> const & inputValues, std::vector<double> const & desiredOutput)
{
	std::vector<double> const & output = CalcOutput(inputValues);
	UpdateWeights(inputValues, output, desiredOutput);


	return output;
}

std::vector<double> MLNN_KandA::ArtificialNN::CalcOutput(std::vector<double> const &  inputValues)
{

	// first input layer preactivation calculation
	for (size_t i = 0; i < numNPerHidden[0]; i++)
	{
		double tempSum = 0.0;
		for (size_t inputVal = 0; inputVal < inputValues.size(); inputVal++)
		{
			tempSum += weights[inputVal + i * numInputs] * inputValues.at(inputVal);
		}
		tempSum += biases[i];

		preActivation[i] = tempSum;
	}

	//	l * neuronsPerHidden = firstLayerSize + (l-1) * neuronsPerHidden

	for (size_t l = 1; l < numNPerHidden.size(); l++)
	{
		//For each neuron in the layer
		for (size_t n = 0; n < numNPerHidden[l]; ++n)
		{
			double tempSum = 0.0;
			//For each input, which means each previous neuron
			size_t numNeuronsPrevious = numNPerHidden[l - 1];
			for (size_t previousN = 0; previousN < numNeuronsPrevious; previousN++) {
				size_t weightOfPreviousN = GetWeightHiddenLayerStartIndex(l) + n * numNeuronsPrevious + previousN;
				size_t previousNeuronPreActivationIndex = GetBiasHiddenLayerStartIndex(l - 1) + previousN;

				tempSum +=
					weights[weightOfPreviousN]
					* Math::ActivationFunction(activationFunctionHiddenLayer[l], preActivation[previousNeuronPreActivationIndex]);

			}
			tempSum += biases[GetBiasHiddenLayerStartIndex(l)];

			size_t preActivationIndex = GetBiasHiddenLayerStartIndex(l) + n;

			preActivation[preActivationIndex] = tempSum;
		}
	}
	//Finished calculating forward
	std::vector<double> outputs(numOutputs, 0.0);

	for (size_t o = 0; o < numOutputs; o++)
	{
		double tempSum = 0.0;
		for (size_t n = 0; n < numNPerHidden.back(); n++)
		{
			size_t weightOfPreviousN = GetWeightOutputStartIndex() + o * numNPerHidden.back() + n;
			size_t lastLayerNumber = numNPerHidden.size() - 1;
			size_t previousNeuronPreActivationIndex = GetBiasHiddenLayerStartIndex(lastLayerNumber) + n;



			tempSum += weights[weightOfPreviousN]
				* Math::ActivationFunction(activationFunctionOutputLayer, preActivation[previousNeuronPreActivationIndex]);
		}

		tempSum += biases[GetBiasOutputStartIndex() + o];

		preActivation[GetBiasOutputStartIndex() + o] = tempSum;
		outputs[o] = Math::ActivationFunction(activationFunctionOutputLayer, preActivation[GetBiasOutputStartIndex() + o]);
	}
	
	return outputs;
}

void MLNN_KandA::ArtificialNN::UpdateWeights(std::vector<double> const &inputValues, 
	std::vector<double> const & outputs, std::vector<double> const & desiredOutput)
{
	std::vector<double> errorGradient(biases.size(), 0.0);
	//calculate error gradient for output
	for (size_t o = 0; o < numOutputs; o++)
	{
		size_t outputIndex = GetBiasOutputStartIndex() + o;
		double calculatedOutput = ActivationFunction(activationFunctionOutputLayer, preActivation[outputIndex]);
		double errorDifference = desiredOutput[o] - calculatedOutput;
		double outputErrorGadient = errorDifference * DerivedFunction(activationFunctionOutputLayer, calculatedOutput);
		errorGradient[outputIndex] = outputErrorGadient;
	}

	//calculate error gradient for hidden last layer
	{
		size_t const lastLayer = numHidden - 1;
		size_t const & numberOfNeurons = numNPerHidden[lastLayer];
		for (size_t n = 0; n < numberOfNeurons; ++n)
		{
			size_t const neuronIndex = GetBiasHiddenLayerStartIndex(lastLayer) + n;
			for (size_t o = 0; o < numOutputs; ++o)
			{
				size_t const outputIndex = GetBiasOutputStartIndex() + o;
				double const & outputErrorGradient = errorGradient[outputIndex];
				size_t const weightIndex = GetWeightOutputStartIndex() + o * numNPerHidden.back() + n; // cache miss
				double const & weight = weights[weightIndex];
				double const errorGradientPart = weight * outputErrorGradient;
				errorGradient[neuronIndex] += errorGradientPart;
			}
			errorGradient[neuronIndex] *= ActivateThenDerive(activationFunctionHiddenLayer[lastLayer], preActivation[neuronIndex]);
		}

	}
	//calculate error gradient for other hidden layers
	if(numHidden > 1)
	{
		for (size_t foo = 0; foo < numHidden - 2; ++foo)
		{
			size_t const l = numHidden - 2 - foo;
			size_t const nextLayer = l + 1;
			size_t const & numberOfNeurons = numNPerHidden[l];
			for (size_t n = 0; n < numberOfNeurons; ++n)
			{
				size_t const neuronIndex = GetBiasHiddenLayerStartIndex(l) + n;
				for (size_t m = 0; m < numOutputs; ++m)
				{
					size_t const nextLayerIndex = GetBiasHiddenLayerStartIndex(nextLayer) + m;
					double const & nextLayerErrorGradient = errorGradient[nextLayerIndex];
					size_t const weightIndex = GetWeightHiddenLayerStartIndex(nextLayer) + m * numNPerHidden[l] + n; // cache miss
					double const & weight = weights[weightIndex];
					double const errorGradientPart = weight * nextLayerErrorGradient;
					errorGradient[neuronIndex] += errorGradientPart;
				}
				errorGradient[neuronIndex] *= ActivateThenDerive(activationFunctionHiddenLayer[l], preActivation[neuronIndex]);
			}
		}
	}

	//Updating Weights and biases
	//First layer
	for (size_t n = 0; n < numNPerHidden[0]; n++)
	{
		for (size_t inputVal = 0; inputVal < inputValues.size(); inputVal++)
		{
			size_t const weightIndex = inputVal + n * numInputs;
			double const deltaWeight = learningRatePerHidden[0] * errorGradient[n] * inputValues[inputVal];
			weights[weightIndex] += deltaWeight;
		}
		biases[n] += learningRatePerHidden[0] * errorGradient[n];
	}

	//Hidden Layer
	for (size_t l = 1; l < numNPerHidden.size(); l++)
	{
		size_t const & numNeuronsPrevious = numNPerHidden[l - 1];
		double const & learningRate = learningRatePerHidden.at(l);
		double const & neuronsInLayer = numNPerHidden[l];
		//For each neuron in the layer
		for (size_t n = 0; n < numNPerHidden[l]; ++n)
		{
			//For each input, which means each previous neuron
			size_t const fooIndex = l * neuronsInLayer + n;
			double& errorGradientValue = errorGradient[fooIndex];

			for (size_t previousN = 0; previousN < numNeuronsPrevious; previousN++) {
				//Jeg forstår ikke funksjonene dine, så jeg tror de er bugget/ikke komplett, bruker inntil videre hard logikk
				//size_t weightOfPreviousN = GetWeightHiddenLayerStartIndex(l) + n * numNeuronsPrevious + previousN;
				//size_t previousNeuronPreActivationIndex = GetBiasHiddenLayerStartIndex(l - 1) + previousN;

				double& weightCurrent = weights[(numInputs * neuronsInLayer) + (l - 1) * neuronsInLayer * neuronsInLayer + n * neuronsInLayer + previousN];
				double& preActivationValue = preActivation[(l - 1) * neuronsInLayer + previousN];

				weights[(numInputs * neuronsInLayer) + (l - 1) * neuronsInLayer * neuronsInLayer + n * neuronsInLayer + previousN]
					= weightCurrent
					+ learningRate
					* errorGradientValue
					* Math::ActivationFunction(activationFunctionHiddenLayer[0], preActivationValue);
			}
			double& biasCurrent = biases[numInputs + (l - 1) * neuronsInLayer + n];

			biases[numInputs + (l - 1) * neuronsInLayer + n]
				= biasCurrent
				+ learningRate
				* errorGradientValue;
		}
	}

	//Output Layer
	for (size_t o = 0; o < numOutputs; o++)
	{
		size_t const biasIndex = GetBiasOutputStartIndex() + o;
		double const & errorGradientValue = errorGradient[biasIndex];
		for (size_t n = 0; n < numNPerHidden.back(); n++)
		{
			size_t const weightIndex = weightOutputStartIndex + o * numNPerHidden.back() + n;

			double const preActivationIndex = (numNPerHidden.size() - 1) * numNPerHidden.back() + n;
			double const & preActivationValue = preActivation[preActivationIndex];

			double const deltaWeight =
				+ learningRateOutput
				* errorGradientValue
				* Math::ActivationFunction(activationFunctionOutputLayer, preActivationValue);

			weights[weightIndex] += deltaWeight;
		}
		

		biases[biasIndex]
			+= learningRateOutput
			* errorGradientValue;
	}
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


