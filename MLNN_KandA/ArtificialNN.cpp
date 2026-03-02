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
	/* The number of weights in each hidden layer and the output layer.
	* Each index is the corresponding hidden layer's index
	* The output layer is the .back() index
	*/
	std::vector<size_t> weightLayerSize;

	//setup hidden layer start indices
	{
		weightLayerSize.resize(numHidden + 1); // resize to 1 index per layer + 1 for output

		weightHiddenLayerStartIndex.push_back(0);
		biasHiddenLayerStartIndex.push_back(0);
		weightLayerSize[0] = numInputs * numNPerHidden[0];
		for (size_t hLayer = 1; hLayer < numNPerHidden.size(); ++hLayer)
		{
			size_t prevIndex = hLayer - 1;
			weightHiddenLayerStartIndex.push_back(weightHiddenLayerStartIndex[prevIndex] + weightLayerSize.at(prevIndex));
			biasHiddenLayerStartIndex.push_back(biasHiddenLayerStartIndex[prevIndex] + GetBiasLayerSize(prevIndex));
			weightLayerSize[hLayer] = numNPerHidden[prevIndex] * numNPerHidden[hLayer];
		}
	}


	//initializing output layer
	{
		size_t backIndex = weightHiddenLayerStartIndex.back();
		size_t sizeOfLastLayer = weightLayerSize.at(numHidden-1);

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
		size_t weightsSize = GetWeightOutputStartIndex() + numOutputs * numNPerHidden.back();
		size_t biasesSize = GetBiasOutputStartIndex() + numOutputs;

		weights.resize(weightsSize);
		biases.resize(biasesSize);
		preActivation.resize(biasesSize, 0.0);

		for (size_t i = 0; i < weightsSize; ++i)
		{
			weights[i] = 0.0; //RNG::GetNumber();
		}
		for (size_t i = 0; i < biasesSize; ++i)
		{
			biases[i] = 0.0; //RNG::GetNumber();
		}
	}
}

std::vector<double> ArtificialNN::Train(std::vector<double> const & inputValues, std::vector<double> const & desiredOutput)
{
	std::vector<double> const & output = CalcOutput(inputValues);
	UpdateWeights(inputValues, output, desiredOutput);


	return output;
}

std::vector<double> MLNN_KandA::ArtificialNN::CalcOutput(std::vector<double> const &  inputValues)
{

	// first input layer preactivation calculation
	for (size_t n = 0; n < numNPerHidden[0]; n++)
	{
		double tempSum = 0.0;
		for (size_t inputIndex = 0; inputIndex < inputValues.size(); inputIndex++)
		{
			size_t const weightIndex = inputIndex + n * numInputs;
			tempSum += weights[weightIndex] * inputValues.at(inputIndex);
		}
		tempSum += biases[n];

		preActivation[n] = tempSum;
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
					* Math::ActivationFunction(activationFunctionHiddenLayer[l-1], preActivation[previousNeuronPreActivationIndex]);

			}
			size_t preActivationIndex = GetBiasHiddenLayerStartIndex(l) + n;
			tempSum += biases[preActivationIndex];


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
				* Math::ActivationFunction(activationFunctionHiddenLayer.back(), preActivation[previousNeuronPreActivationIndex]);
		}
		size_t const preActivationIndex = GetBiasOutputStartIndex() + o;
		tempSum += biases[preActivationIndex];

		preActivation[preActivationIndex] = tempSum;
		outputs[o] = Math::ActivationFunction(activationFunctionOutputLayer, preActivation[preActivationIndex]);
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
		double calculatedOutput = outputs[o];
		double errorDifference = desiredOutput[o] - calculatedOutput;

		double const & preActivationValue = preActivation[GetBiasOutputStartIndex()+o];

		double outputErrorGadient = errorDifference * DerivedFunction(activationFunctionOutputLayer, preActivationValue);
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
			errorGradient[neuronIndex] *= DerivedFunction(activationFunctionHiddenLayer[lastLayer], preActivation[neuronIndex]);
		}

	}

	//calculate error gradient for other hidden layers
	if(numHidden > 1)
	{
		for (size_t temp = 0; temp < numHidden - 2; ++temp)
		{
			size_t const l = numHidden - 2 - temp;
			size_t const nextLayer = l + 1;
			size_t const & numberOfNeurons = numNPerHidden[l];
			for (size_t n = 0; n < numberOfNeurons; ++n)
			{
				size_t const neuronIndex = GetBiasHiddenLayerStartIndex(l) + n;
				for (size_t m = 0; m < numNPerHidden[nextLayer]; ++m)
				{
					size_t const nextLayerIndex = GetBiasHiddenLayerStartIndex(nextLayer) + m;
					double const & nextLayerErrorGradient = errorGradient[nextLayerIndex];
					size_t const weightIndex = GetWeightHiddenLayerStartIndex(nextLayer) + m * numNPerHidden[l] + n; // cache miss
					double const & weight = weights[weightIndex];
					double const errorGradientPart = weight * nextLayerErrorGradient;
					errorGradient[neuronIndex] += errorGradientPart;
				}
				errorGradient[neuronIndex] *= DerivedFunction(activationFunctionHiddenLayer[l], preActivation[neuronIndex]);
			}
		}
	}


	//Updating Weights and biases

	//Output Layer
	for (size_t o = 0; o < numOutputs; o++)
	{
		size_t const finalLayerIndex = numHidden - 1;
		size_t const biasIndex = GetBiasOutputStartIndex() + o;
		double const& errorGradientValue = errorGradient[biasIndex];

		for (size_t n = 0; n < numNPerHidden.back(); n++)
		{
			size_t const weightIndex = GetWeightOutputStartIndex() + o * numNPerHidden.back() + n;

			size_t const preActivationIndex = GetBiasHiddenLayerStartIndex(finalLayerIndex) + n;
			double const& preActivationValue = preActivation[preActivationIndex];

			double const deltaWeight =
				learningRateOutput
				* errorGradientValue
				* Math::ActivationFunction(activationFunctionOutputLayer, preActivationValue);

			weights[weightIndex] += deltaWeight;
		}


		biases[biasIndex] +=
			learningRateOutput
			* errorGradientValue;
	}

	//Hidden Layer
	for (size_t l = 1; l < numNPerHidden.size(); l++)
	{
		size_t const& numNeuronsPrevious = numNPerHidden[l - 1];
		double const& learningRate = learningRatePerHidden.at(l);
		size_t const& neuronsInLayer = numNPerHidden[l];
		//For each neuron in the layer
		for (size_t n = 0; n < numNPerHidden[l]; ++n)
		{
			//For each input, which means each previous neuron
			size_t const biasIndex = GetBiasHiddenLayerStartIndex(l) + n;
			double& errorGradientValue = errorGradient[biasIndex];

			for (size_t previousN = 0; previousN < numNeuronsPrevious; previousN++) {

				size_t const weightIndex = GetWeightHiddenLayerStartIndex(l) + previousN + n * numNeuronsPrevious;

				size_t const preActivationIndex = GetBiasHiddenLayerStartIndex(l - 1) + previousN;
				double& preActivationValue = preActivation[preActivationIndex];

				weights[weightIndex] +=
					learningRate
					* errorGradientValue
					* Math::ActivationFunction(activationFunctionHiddenLayer[0], preActivationValue);
			}

			biases[biasIndex] +=
				learningRate
				* errorGradientValue;
		}
	}

	//First layer
	for (size_t n = 0; n < numNPerHidden[0]; n++)
	{
		for (size_t inputIndex = 0; inputIndex < inputValues.size(); inputIndex++)
		{
			size_t const weightIndex = inputIndex + n * numInputs;
			double const deltaWeight = learningRatePerHidden[0] * errorGradient[n] * inputValues[inputIndex];
			weights[weightIndex] += deltaWeight;
		}
		biases[n] += learningRatePerHidden[0] * errorGradient[n];
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


