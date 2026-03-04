#pragma once
#include "ArtificialNN.h"
#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>

namespace MLNN_KandA
{

class Replay
{
public:
	std::vector<double> states;
	double reward;

	Replay(std::vector<double> inputs, double r)
	{
		for (auto& elm : inputs)
		{
			states.push_back(elm);
		}
		reward = r;
	}
};

class Q_Learning
{
private:
	std::shared_ptr<ArtificialNN> ann;

	std::vector<std::shared_ptr<Replay>> replayMemory;			//memory - list of past actions and rewards
	int mCapacity = 10000;							//memory capacity

	float discount = 0.99f;							//how much future states affect rewards

public:
	Q_Learning(int mCapacity, float discount, std::shared_ptr<ArtificialNN> ann);

	static std::vector<double>& SoftMax(std::vector<double>& values)
	{
		double sum = std::accumulate(values.begin(), values.end(), 0.0);
		if(sum != 0.0)
		{
			std::for_each(values.begin(), values.end(),
				[&sum](double& n) { n/=sum; }
			);
		}
		return values;
	};

	void AddReplayMemory(std::vector<double> states, double reward);

	void ProcessQLearningWithTrainANN();

};
}