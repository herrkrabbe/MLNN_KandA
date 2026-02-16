#include <iostream>
#include "MLP_Library/RNG.h"
#include "MLP_Library/RNG.cpp"

int main()
{
	std::cout << "hello world" << std::endl;
	double number = RNG::GetNumber();
	std::cout << number << std::endl;
}