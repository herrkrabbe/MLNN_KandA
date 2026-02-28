#include <iostream>
#include "MLNN_KandA/RNG.h"
#include "MLNN_KandA/ActivationFunctions.h"
#include "MLNN_KandA/ArtificialNN.h"

int main()
{
	std::cout << "hello world" << std::endl;
	double number = RNG::GetNumber();
	std::cout << number << std::endl;
	double result = MLNN_KandA::Math::ActivationFunction(MLNN_KandA::Math::eActivationFunction::Identity, number);
	std::cout << result << std::endl;

	MLNN_KandA::ArtificialNN foo(2, 1, 2, 2, 0.0001, 0.0001,
		MLNN_KandA::Math::eActivationFunction::Sigmoid, MLNN_KandA::Math::eActivationFunction::Sigmoid);
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

	for(int i = 0; i<4; ++i)
	{
		foo.Train(inputs[i], outputs[i]);
	}
}