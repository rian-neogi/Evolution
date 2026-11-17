#include "Environment.h"

Environment::Environment() : mGame(nullptr), mPopulation(nullptr), mPopulationSize(0) {}

Environment::Environment(Game* game, Genotype* pop, int pop_size) : Environment()
{
    init(game, pop, pop_size);
}

Environment::~Environment() {}

void Environment::init(Game* game, Genotype* pop, int pop_size)
{
    mGame = game;
    mPopulation = pop;
    mPopulationSize = pop_size;
    // Keep the legacy ownership convention: callers own the interpreter's buffers.
    if (!mExe.mRegisters)
        mExe.mRegisters = new int[8];
    mExe.mRegisterSize = 8;
    mExe.mInputSize = mGame->getStateSize();
    mExe.mInputs = mGame->getState();
    mExe.reset();
}

int Environment::getFitness(int indi)
{
    mGame->restart();
    int totalReward = 0;
    const int actions = mGame->getActionCount();
    for (int frame = 0; frame < mMaxFrames && !mGame->is_over(); ++frame)
    {
        mExe.reset();
        mExe.mInputs = mGame->getState();
        mExe.run_code(&mPopulation[indi], 0);
        const int action = (mExe.mRegisters[0] % actions + actions) % actions;
        totalReward += mGame->do_action(action);
    }
    return totalReward;
}
