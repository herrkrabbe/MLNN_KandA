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

	MLNN_KandA::ArtificialNN foo(4, 4, 4, 5, 0.0001, 0.0001,
		MLNN_KandA::Math::eActivationFunction::Sigmoid, MLNN_KandA::Math::eActivationFunction::Sigmoid);
	foo.PrintLayerIndices();
}