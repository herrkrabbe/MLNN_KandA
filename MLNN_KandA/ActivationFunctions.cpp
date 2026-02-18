#include "ActivationFunctions.h"

double MLNN_KandA::Math::ActivationFunction(eActivationFunction const& f, double const& x)
{
	switch (f) {
	case eActivationFunction::Identity:
		return funcs::ActivationIdentity(x);
		break;
	case eActivationFunction::BinaryStep:
		return funcs::ActivationBinaryStep(x);
		break;
	case eActivationFunction::Sigmoid:
		return funcs::ActivationSigmoid(x);
		break;
	case eActivationFunction::ReLu:
		return funcs::ActivationReLU(x);
		break;
	case eActivationFunction::LeakyReLU:
		return funcs::ActivationLeakyReLU(x);
		break;
	default:
		throw("Activation function lacks an implementation");
		return 0.0;
	}

}

double MLNN_KandA::Math::DerivedFunction(eActivationFunction const& f, double const& x)
{
	switch (f) {
	case eActivationFunction::Identity:
		return funcs::DerivedIdentity(x);
		break;
	case eActivationFunction::BinaryStep:
		return funcs::DerivedBinaryStep(x);
		break;
	case eActivationFunction::Sigmoid:
		return funcs::DerivedSigmoid(x);
		break;
	case eActivationFunction::ReLu:
		return funcs::DerivedReLU(x);
		break;
	case eActivationFunction::LeakyReLU:
		return funcs::DerivedLeakyReLU(x);
		break;
	default:
		throw("Derived activation function lacks an implementation for value");
		return 0.0;
	}

}

double MLNN_KandA::Math::MSE(std::vector<double> a, std::vector<double> b)
{
	bool rightSize = a.size() == b.size();
	if (!rightSize)
	{
		std::cout << "WARNING: mean square error input vectors are different size" << "\n";
		throw;
	}

	double mse{ 0.0 };
	for (int i{ 0 }; i < a.size(); ++i)
	{
		mse += MSE(a[i], b[i]);
	}
	return mse;
}
