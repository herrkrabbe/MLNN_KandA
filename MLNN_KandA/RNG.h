#pragma once

#include <random>
#include <iostream>
#include <functional>

//int main() {
//	std::random_device rd;          // non-deterministic seed
//	std::mt19937 gen(rd());         // Mersenne Twister engine
//	std::uniform_int_distribution<> dist(1, 10);
//
//	int value = dist(gen);
//	std::cout << value << "\n";
//}


namespace RNG{
	//class RNG {
	//	bool isSeeded{false};
	//	std::mt19937 gen;         // Mersenne Twister engine
	//	std::uniform_real_distribution<> dist;
	//	//std::_Binder<std::_Unforced, std::uniform_real_distribution<>&, std::mt19937&> rng;
	//public:
	//	void TrySeed();
	//	//Get random number between -1 and 1
	//	double GetNumber();
	//};

	//extern RNG rng;
	//static std::random_device rd;
	//static std::mt19937 gen;         // Mersenne Twister engine
	//static std::uniform_real_distribution<> dist;

	/*static double GetNumber() ;*/
	static std::random_device rd;
	static std::mt19937 gen = std::mt19937(rd());
	static std::uniform_real_distribution<> dist = std::uniform_real_distribution<>(-1.0, 1.0);

	static double GetNumber()
	{
		return dist(gen);
	}
}


