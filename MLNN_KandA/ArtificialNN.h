#pragma once

#include <string>
#include <vector>
#include "ActivationFunctions.h"

/**
 *
 */
namespace MLNN_KandA {
	class ArtificialNN
	{
	public:

		std::vector<double> weights;
		std::vector<double> biases;

		const size_t numInputs;
		const size_t numOutputs;
		//int numHidden;		Instead use		=>	 numNPerHidden.size()
		const std::vector<size_t> numNPerHidden;
		#define numHidden numNPerHidden.size()

		const double learningRateOutput;
		const std::vector<double> learningRatePerHidden;

		//Identical layer amount variant
		ArtificialNN(size_t numberInput, size_t numberOutput,
			size_t numberHiddenLayer, size_t numberNeuronHiddenLayer, double OutputLearningRate, double learningRate,
			Math::eActivationFunction af_HiddenLayer, Math::eActivationFunction af_OutputLayer);

		//Varied Neurons per hidden layer variant
		ArtificialNN(size_t numberInput, size_t numberOutput,
			std::vector<size_t> numberNeuronPerHiddenLayer,
			double OutputLearningRate, std::vector<double> learningRatePerHiddenLayer,
			std::vector<Math::eActivationFunction> af_PerHiddenLayer, Math::eActivationFunction af_OutputLayer);

		~ArtificialNN() = default;
	private:
		void Init();
	public:

		//Train method is to compute the output + update weight
		std::vector<double> Train(std::vector<double> inputValues, std::vector<double> desiredOutput);

		//CalcOutput method is only to compute the output (without update weight for training)
		std::vector<double> CalcOutput(std::vector<double> inputValues);

		std::string PrintWeightsBias();

		// Saves the weights and bias of the current network to a file (not implemented here)
		void SaveWeightsBias();
		// Loads the weights and bias of the current network from a file (not implemented here)
		void LoadWeightsBias();

		void PrintLayerIndices();

	private:
		std::vector<size_t> weightHiddenLayerStartIndex;
		size_t weightOutputStartIndex;

		std::vector<size_t> biasHiddenLayerStartIndex;
		size_t biasOutputStartIndex;

		std::vector<size_t> weightLayerSize;

		const std::vector<Math::eActivationFunction> activationFunctionHiddenLayer;
		const Math::eActivationFunction activationFunctionOutputLayer;

		std::vector<double> preActivation;

		// perform backpropagation to update weights
		void UpdateWeights(std::vector<double> inputValues, std::vector<double> outputs, std::vector<double> desiredOutput);

		//first hidden layer is layer = 0

		// zero indexed
		size_t GetWeightHiddenLayerStartIndex(size_t const& layerIndex)
		{
			return weightHiddenLayerStartIndex.at(layerIndex);
		}

		inline size_t GetWeightOutputStartIndex()
		{
			return weightOutputStartIndex;
		}

		//zero indexed
		inline size_t GetBiasHiddenLayerStartIndex(size_t const & layerIndex)
		{
			return biasHiddenLayerStartIndex.at(layerIndex);
		}

		inline size_t GetBiasOutputStartIndex()
		{
			return biasOutputStartIndex;
		}

		/*
		* Return the size of layer.
		* if layerIndex == number of hidden layers, it will return the size of the
		* output layer. That is because layersIndex is zero indexed
		*/
		inline size_t GetWeightLayerSize(size_t const& layerIndex) const
		{
			return weightLayerSize.at(layerIndex);
		}

		size_t GetBiasLayerSize(size_t const& layerIndex) const
		{
			if(layerIndex == numNPerHidden.size()) // output layer index
			{
				return numOutputs;
			}
			else
			{
				return numNPerHidden.at(layerIndex);
			}
		}

		
	};
}
