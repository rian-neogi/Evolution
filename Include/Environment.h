#pragma once

#include "Game.h"

class Environment 
{
public:
	Game* mGame;
	Execution mExe;
	Genotype* mPopulation;
	int mPopulationSize;
	int mMaxFrames = 10001;

	Environment();
	Environment(Game* game, Genotype* pop, int pop_size);
	virtual ~Environment();

	virtual void init(Game* game, Genotype* pop, int pop_size);
	virtual int getFitness(int indi);
};
