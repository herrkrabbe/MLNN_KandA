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

		int numInputs;
		int numOutputs;
		//int numHidden;		Instead use		=>	 numNPerHidden.size()
		std::vector<int> numNPerHidden;

		double learningRateOutput;
		std::vector<double> learningRatePerHidden;

		//Identical layer amount variant
		ArtificialNN(int numberInput, int numberOutput,
			int numberHiddenLayer, int numberNeuronHiddenLayer, int OutputLearningRate, double learningRate,
			Math::eActivationFunction af_HiddenLayer, Math::eActivationFunction af_OutputLayer);

		//Varied Neurons per hidden layer variant
		ArtificialNN(int numberInput, int numberOutput,
			int numberHiddenLayer, std::vector<int> numberNeuronPerHiddenLayer,
			int OutputLearningRate, std::vector<double> learningRatePerHiddenLayer,
			std::vector<Math::eActivationFunction> af_PerHiddenLayer, Math::eActivationFunction af_OutputLayer);

		~ArtificialNN() = default;

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

		std::vector<Math::eActivationFunction> activationFunctionHiddenLayer;
		Math::eActivationFunction activationFunctionOutputLayer;

		// perform backpropagation to update weights
		void UpdateWeights(std::vector<double> outputs, std::vector<double> desiredOutput);


		double ActivationFunctionH(double value);	//hidden layer
		double ActivationFunctionO(double value);	//output layer

		double Step(double value);

		double TanH(double value);

		double Sigmoid(double value);

		double ReLU(double value);

		double LeakyReLU(double value);

		double Derivated_Activation_Function(Math::eActivationFunction af, double value);

		//first hidden layer is layer = 0
		static int GetHiddenLayerStartIndex(int layer, int const& numInputs, int const& numberNeuronHiddenLayer, int const& numberHiddenLayer)
		{
			if(layer <= 0)
			{
				return 0;
			}
			if(layer > numberHiddenLayer-1)
			{
				layer = numberHiddenLayer;
			}
			return numInputs * numberHiddenLayer + (layer-1)* numberNeuronHiddenLayer * numberNeuronHiddenLayer;
		}
		static inline int GetOutputLayerStartIndex(int const & numInputs, int const & numberNeuronHiddenLayer, int const & numberHiddenLayer)
		{
			return numInputs * numberNeuronHiddenLayer + (numberHiddenLayer - 1) * numberNeuronHiddenLayer * numberNeuronHiddenLayer;
		}
	};
}
