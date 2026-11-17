#include "SupervisedEnvironment.h"

SupervisedEnvironment::SupervisedEnvironment()
{

}

SupervisedEnvironment::SupervisedEnvironment(AtariGame* game, Genotype* pop, int pop_size, Genotype* supervisor) 
	: Environment(game, pop, pop_size), mSupervisor(supervisor)
{
	mScores = new int[pop_size];
	memset(mScores, 0, pop_size*sizeof(int));
}

SupervisedEnvironment::~SupervisedEnvironment()
{
	// if(mScores!=NULL)
		// delete [] mScores;
}

int SupervisedEnvironment::getFitness(int indi)
{
	auto* game = static_cast<AtariGame*>(mGame);
	mGame->restart();
	mExe.reset();
	int gameReward = 0;
	int supervisorReward = 0;
	while (!mGame->is_over())
	{
		gameReward += game->mALE->act(PLAYER_A_FIRE);
		mExe.reset();
		mExe.run_code(&mPopulation[indi], 0);

		ActionVect vect = game->mALE->getMinimalActionSet();
		int size = vect.size();
		gameReward += mGame->do_action(mExe.mRegisters[0] % size);

		mExe.reset();
		mExe.run_code(mSupervisor, 0);
		supervisorReward += mExe.mRegisters[0];
	}
	mScores[indi] = gameReward;
	return supervisorReward;
}
