#include "Q_Learning.h"

MLNN_KandA::Q_Learning::Q_Learning(size_t _mCapacity, float _discount, std::shared_ptr<ArtificialNN> _ann)
	:
	discount(_discount),
	ann(_ann)
{
	replayMemory.resize(_mCapacity);
}

void MLNN_KandA::Q_Learning::AddReplayMemory(std::vector<double> const& states, double const& reward)
{
	if(states.size() == 0) return;
	if(currentMemoryIndex >= replayMemory.size())
	{
		currentMemoryIndex = 0;
	}

	replayMemory[currentMemoryIndex] = std::make_unique<Replay>(states, reward);

	++currentMemoryIndex;
}

void MLNN_KandA::Q_Learning::ProcessQLearningWithTrainANN()
{
}
