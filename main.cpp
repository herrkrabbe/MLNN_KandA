#include <iostream>
#include "MLNN_KandA/RNG.h"
#include "MLNN_KandA/ActivationFunctions.h"
#include "MLNN_KandA/ArtificialNN.h"
#include <string>
#include <sstream>

int main()
{
	std::cout << "hello world" << std::endl;

	size_t numInputs = 2;
	size_t numOutputs = 1;
	size_t numHiddenLayers = 1;
	size_t numNeuronsInHiddenLayer = 2;
	double outputLayerLearningRate = 0.1;
	double hiddenLayerLearningRate = 0.1;
	MLNN_KandA::Math::eActivationFunction hiddenActivationFunction = MLNN_KandA::Math::eActivationFunction::Sigmoid;
	MLNN_KandA::Math::eActivationFunction outputActivationFunction = MLNN_KandA::Math::eActivationFunction::Sigmoid;


	MLNN_KandA::ArtificialNN foo(numInputs, numOutputs, numHiddenLayers, numNeuronsInHiddenLayer
		, outputLayerLearningRate, hiddenLayerLearningRate,
		hiddenActivationFunction, outputActivationFunction);
	foo.PrintLayerIndices();

	std::vector<std::vector<double>> inputs
	{
		{0.0, 0.0}
		,{0.0, 1.0}
		,{1.0, 0.0}
		,{1.0, 1.0}
	};
	std::vector<std::vector<double>> outputs
	{
		{0.0}
		,{1.0}
		,{1.0}
		,{0.0}
	};
	for(int i = 0; i < 10000; ++i)
	{
		for(int i = 0; i<4; ++i)
		{
			foo.Train(inputs[i], outputs[i]);
		}
	}
	std::vector<std::vector<double>> results;
	std::stringstream ss;
	for(int i = 0; i < 4; ++i)
	{
		std::vector<double> result = foo.CalcOutput(inputs[i]);
		results.push_back(result);

		ss << inputs[i][0] << " " << inputs[i][1] << " -> " << result[0] << std::endl;
	}
	std::cout << ss.str();

}