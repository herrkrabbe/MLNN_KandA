#pragma once
#include "ArtificialNN.h"
#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace MLNN_KandA
{

class Replay
{
public:
	std::vector<double> states;
	double reward;

	Replay(std::vector<double> inputs, double r)
	{
		states = inputs;
		reward = r;
	}
};

class Q_Learning
{
private:
	std::vector<std::unique_ptr<Replay>> replayMemory;			//memory - list of past actions and rewards

	float discount = 0.99f;							//how much future states affect rewards

	std::shared_ptr<ArtificialNN> ann;

	size_t currentMemoryIndex = 0;

public:
	Q_Learning(size_t _mCapacity, float _discount, std::shared_ptr<ArtificialNN> _ann);

	static std::vector<double> SoftMax(std::vector<double> values, double temperature = 1.0)
	{
		if(temperature != 0.0)
		{
			temperature = abs(temperature);
			double maxValue = *std::max_element(values.begin(), values.end());

			std::for_each(values.begin(), values.end(),
				[&maxValue, &temperature](double& n) { n = exp(n - maxValue); }
			);
			double exponentialSum = std::accumulate(values.begin(), values.end(), 0.0);
			std::for_each(values.begin(), values.end(),
				[exponentialSum](double& n) { n /= exponentialSum; }
			);
		}
		return values;
	};

	void AddReplayMemory(std::vector<double> const & states, double const & reward);

	void ProcessQLearningWithTrainANN();

};
}