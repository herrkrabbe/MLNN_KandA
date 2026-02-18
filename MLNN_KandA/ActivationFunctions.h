#pragma once
#include <cmath>
#include <cassert>
#include <iostream>

namespace MLNN_KandA
{
namespace Math
{
	enum eActivationFunction
	{
		Identity = 0
		, BinaryStep = 1
		, Sigmoid = 2
		, ReLu = 3
		, LeakyReLU = 4
	};

	namespace funcs
	{
		inline double ActivationIdentity(double const& x)
		{
			return x;
		}
		inline double ActivationBinaryStep(double const& x)
		{
			return x >= 0.0;
		}
		inline double ActivationSigmoid(double const & x)
		{
			return 1.0/(1.0 + std::exp(-x));
		}
		inline double ActivationReLU(double const& x)
		{
			return (x >= 0.0) ? x : 0.0;
		}
		inline double ActivationLeakyReLU(double const& x)
		{
			return (x >= 0.0) ? x : 0.01*x;
		}
		inline double DerivedIdentity(double const& x)
		{
			return 1.0;
		}
		inline double DerivedBinaryStep(double const& x)
		{
			return 0.0;
		}
		inline double DerivedSigmoid(double const& x)
		{
			double sig = ActivationSigmoid(x);
			return sig * (1.0 - sig);
		}
		inline double DerivedReLU(double const& x)
		{
			return (x > 0) ? 1.0 : 0.0;
		}
		inline double DerivedLeakyReLU(double const& x)
		{
			return (x > 0) ? 1.0 : 0.01;
		}
	}
	
	double ActivationFunction(eActivationFunction const& f, double const& x)
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

	double DerivedFunction(eActivationFunction const& f, double const& x)
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

	inline double MSE(double const& a, double const& b)
	{
		return pow(a*b, 2.0);
	}

	double MSE(std::vector<double> a, std::vector<double> b)
	{
		bool rightSize = a.size() == b.size();
		if(!rightSize)
		{
			std::cout << "WARNING: mean square error input vectors are different size" << "\n";
			throw;
		}
		
		double mse{0.0};
		for(int i{0}; i<a.size(); ++i)
		{
			mse += MSE(a[i], b[i]);
		}
		return mse;
	}
}
}
	
