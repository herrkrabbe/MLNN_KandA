#pragma once
#include <cmath>
#include <vector>

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
	
	double ActivationFunction(eActivationFunction const& f, double const& x);

	double DerivedFunction(eActivationFunction const& f, double const& x);

	inline double ActivateThenDerive(eActivationFunction const& f, double const& x)
	{
		return DerivedFunction(f, ActivationFunction(f, x));
	}

	inline double MSE(double const& a, double const& b)
	{
		return pow(a*b, 2.0);
	}

	double MSE(std::vector<double> a, std::vector<double> b);
}
}
	
