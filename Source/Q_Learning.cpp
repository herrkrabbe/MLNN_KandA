#include "Q_Learning.h"

MLNN_KandA::Q_Learning::Q_Learning(size_t _mCapacity, double _discount, std::shared_ptr<ArtificialNN> _ann)
	:
	discount(_discount),
	ann(_ann)
{
	replayMemory.resize(_mCapacity);
}

void MLNN_KandA::Q_Learning::AddReplayMemory(std::vector<double> const& states, double const& reward)
{
	if(states.size() == 0) return;
	if(currentMemoryIndex == size_t(0)-1)
	{
		return;
	}

	replayMemory[currentMemoryIndex] = std::make_unique<Replay>(states, reward);

	++currentMemoryIndex;
}

void MLNN_KandA::Q_Learning::ProcessQLearningWithTrainANN()
{
	for (auto iter = replayMemory.rbegin(); iter != replayMemory.rend()-1; ++iter)
	{
		auto& M = (*iter);
		auto& MPlusOne = *(iter + 1);
		std::vector<double> oldQMax = SoftMax(ann->CalcOutput(M->states));
		std::vector<double> newQMax = SoftMax(ann->CalcOutput(MPlusOne->states));
		//size_t oldAction = std::max_element(oldQMax.begin(), oldQMax.end()) - oldQMax.begin();
		double feedback = M->reward + discount * newQMax[0];

		ann->Train(M->states, {feedback});
		M.reset();
	}
	(*(replayMemory.rend() - 1)).reset();
	currentMemoryIndex = 0;
}
