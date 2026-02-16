#include "RNG.h"

//void MLP_Library::RNG::TrySeed()
//{
//	if(isSeeded) return;
//	std::random_device rd;          // non-deterministic seed
//	gen = std::mt19937(rd());         // Mersenne Twister engine
//	dist = std::uniform_real_distribution<>(-1.0, 1.0);
//
//	double value = dist(gen);
//	std::cout << value << "\n";
//}
//
//double MLP_Library::RNG::GetNumber()
//{
//	TrySeed();
//	return dist(gen);
//}
