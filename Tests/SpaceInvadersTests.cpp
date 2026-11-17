#include "LocalTraining.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

struct SpaceInvadersTestAccess
{
    static SpaceInvaders::World& world(SpaceInvaders& game) { return game.mWorld; }
};

static void check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

static void replay()
{
    SpaceInvaders a(42, 1800), b(42, 1800);
    std::vector<std::array<byte_t, 128>> states;
    std::vector<int> rewards;
    int total = 0;
    for (int frame = 0; frame < 1800; ++frame)
    {
        const int action = (frame / 23 + frame % 7) % 6;
        const int reward = a.do_action(action);
        check(reward == b.do_action(action), "Same seed/actions must give identical rewards");
        // Observation frequency and renderer-style world reads must not affect simulation.
        b.getState(); b.getState(); (void)b.world();
        check(std::memcmp(a.getState(), b.getState(), 128) == 0, "Deterministic state replay failed");
        std::array<byte_t, 128> state;
        std::memcpy(state.data(), a.getState(), state.size());
        states.push_back(state);
        rewards.push_back(reward);
        total += reward;
    }
    check(total == a.world().score, "Rewards must sum to displayed score");
    a.restart();
    for (int frame = 0; frame < 1800; ++frame)
    {
        check(a.do_action((frame / 23 + frame % 7) % 6) == rewards[frame], "Reset did not restore RNG/rewards");
        check(std::memcmp(a.getState(), states[frame].data(), 128) == 0, "Reset did not restore complete state");
    }
    SpaceInvaders other(43, 1800);
    b.restart();
    bool different = false;
    for (int frame = 0; frame < 100; ++frame)
    {
        other.do_action(SpaceInvaders::Noop);
        b.do_action(SpaceInvaders::Noop);
        different |= std::memcmp(other.getState(), b.getState(), 128) != 0;
    }
    check(different, "Different seeds should produce different enemy fire");
}

static void collisions()
{
    SpaceInvaders game;
    auto& w = SpaceInvadersTestAccess::world(game);
    w.playerShot.active = true;
    w.playerShot.x = SpaceInvaders::enemyX(w, 0);
    w.playerShot.y = SpaceInvaders::enemyY(w, 4) + 7;
    check(game.do_action(SpaceInvaders::Noop) == 10, "Bullet should kill the bottom invader");
    check(w.remaining == 54 && !(w.invaders[4] & 1), "Hit must remove exactly one invader");
    check(!w.playerShot.active, "Hit must consume the bullet");

    game.restart();
    w.playerShot.active = true;
    w.playerShot.x = SpaceInvaders::shieldX(12) + 1;
    w.playerShot.y = 201;
    game.do_action(SpaceInvaders::Noop);
    check(w.shields[12] == 1 && !w.playerShot.active, "Player bullet must erode a shield cell");
    w.bullets[0].active = true;
    w.bullets[0].x = SpaceInvaders::shieldX(0) + 1;
    w.bullets[0].y = 184;
    game.do_action(SpaceInvaders::Noop);
    check(w.shields[0] == 1 && !w.bullets[0].active, "Enemy bullet must erode a shield cell");

    game.restart();
    w.bullets[0].active = true;
    w.bullets[0].x = w.playerX;
    w.bullets[0].y = SpaceInvaders::PlayerY - 6;
    game.do_action(SpaceInvaders::Noop);
    check(w.lives == 2 && w.invulnerable > 0, "Ship hit must lose one life and give respawn protection");
    w.bullets[0].active = true;
    w.bullets[0].x = w.playerX;
    w.bullets[0].y = SpaceInvaders::PlayerY - 6;
    game.do_action(SpaceInvaders::Noop);
    check(w.lives == 2, "Respawn protection must prevent repeated damage");
    w.invulnerable = 0;
    w.lives = 1;
    w.bullets[0].y = SpaceInvaders::PlayerY - 6;
    game.do_action(SpaceInvaders::Noop);
    check(game.is_over() && w.lives == 0, "Last life lost must terminate the episode");
}

static void wavesAndLimits()
{
    SpaceInvaders game(123, 1000);
    auto& w = SpaceInvadersTestAccess::world(game);
    w.invaders.fill(0);
    w.invaders[0] = 1;
    w.remaining = 1;
    w.playerShot.active = true;
    w.playerShot.x = w.formationX;
    w.playerShot.y = w.formationY + 7;
    check(game.do_action(SpaceInvaders::Noop) == 130, "Final alien must award kill and wave bonus");
    check(w.wave == 2 && w.remaining == 55 && !game.is_over(), "Clearing wave must start next formation");
    for (int i = 0; i < 100; ++i) game.do_action(SpaceInvaders::Left);
    check(w.playerX == 8, "Ship left boundary failed");
    for (int i = 0; i < 100; ++i) game.do_action(SpaceInvaders::Right);
    check(w.playerX == 248, "Ship right boundary failed");

    game.restart();
    w.formationY = SpaceInvaders::PlayerY;
    w.moveTimer = 100;
    game.do_action(SpaceInvaders::Noop);
    check(game.is_over(), "Invader landing must terminate the episode");

    SpaceInvaders bounded(123, 2);
    bounded.do_action(SpaceInvaders::Fire);
    bounded.do_action(SpaceInvaders::Fire);
    check(bounded.is_over() && bounded.world().frame == 2, "Episode frame cap failed");
    std::array<byte_t, 128> state;
    std::memcpy(state.data(), bounded.getState(), state.size());
    check(bounded.do_action(-50) == 0, "Terminal state must not award rewards");
    check(std::memcmp(state.data(), bounded.getState(), 128) == 0, "Terminal state must remain unchanged");
    bounded.restart();
    check(!bounded.is_over() && bounded.world().score == 0 && bounded.world().frame == 0, "Restart failed");
}

static void interpreterAndAgent()
{
    Genotype genotype;
    auto& code = genotype.mGenes[0].mCode;
    code.clear();
    code.push_back(Instruction(INST_SET, 0, SpaceInvaders::LeftFire));
    SpaceInvaders game;
    InvadersAgent agent(genotype);
    check(agent.action(game) == SpaceInvaders::LeftFire, "Saved genotype action mapping failed");
    game.do_action(agent.action(game));
    check(game.world().playerX == 125 && game.world().playerShot.active, "Combined movement/fire failed");
    int registers[8] = {};
    Execution execution(registers, 8, 128);
    execution.mInputs = game.getState();
    code.clear();
    code.push_back(Instruction(INST_SET, 0, std::numeric_limits<int>::max()));
    code.push_back(Instruction(INST_INC, 0, 1));
    execution.run_code(&genotype, 0);
    check(registers[0] == std::numeric_limits<int>::max(), "Agent register overflow must saturate");
}

static void functionCalls()
{
    Genotype genotype;
    genotype.mGenes.resize(3);
    genotype.mGenes[0].mCode = {Instruction(INST_FUNC, 2, 0)};
    genotype.mGenes[1].mCode = {Instruction(INST_SET, 0, SpaceInvaders::LeftFire)};
    genotype.mGenes[2].mCode = {Instruction(INST_SET, 0, SpaceInvaders::RightFire)};
    SpaceInvaders game;
    InvadersAgent agent(genotype);
    check(agent.action(game) == SpaceInvaders::RightFire,
          "A short caller must be able to call any valid gene");

    genotype.mGenes[0].mCode = {Instruction(INST_FUNC, 1, 0), Instruction(INST_INC, 0, 1)};
    check(agent.action(game) == SpaceInvaders::RightFire,
          "Calls must share registers and resume in the caller");

    genotype.mGenes[0].mCode = {Instruction(INST_FUNC, std::numeric_limits<int>::max(), 0)};
    check(agent.action(game) == SpaceInvaders::RightFire,
          "An oversized call target must clamp to the last gene");

    // Mutations can produce cycles. Calls must share a single execution budget.
    genotype.mGenes[0].mCode = {Instruction(INST_FUNC, 1, 0)};
    genotype.mGenes[1].mCode = {Instruction(INST_FUNC, 0, 0)};
    int registers[8] = {};
    Execution execution(registers, 8, 128, 20);
    execution.mInputs = game.getState();
    execution.run_code(&genotype, 0);
    check(execution.mCurrentStep == 20, "Recursive calls must stop at the shared step limit");
}

static void plateauSelection()
{
    // Index zero represents the incumbent with --elite 1. On a flat landscape,
    // neutral offspring must be eligible to take its place as a breeding parent.
    std::vector<long long> scores(16, 100);
    std::vector<int> order, replayOrder;
    std::mt19937 random(42), replayRandom(42);
    std::array<bool, 16> selected{};
    for (int generation = 0; generation < 128; ++generation)
    {
        rankInvadersPopulation(scores, order, random);
        rankInvadersPopulation(scores, replayOrder, replayRandom);
        check(order == replayOrder, "Tie-breaking must replay with the same seed");
        selected[order.front()] = true;
        auto indices = order;
        std::sort(indices.begin(), indices.end());
        for (int i = 0; i < 16; ++i)
            check(indices[i] == i, "Ranking must include every agent exactly once");
    }
    check(std::count(selected.begin(), selected.end(), true) > 1,
          "Equal-fitness offspring must be able to replace the incumbent parent");

    scores = {10, 40, 40, 5, 100, 40, 10};
    for (int generation = 0; generation < 32; ++generation)
    {
        rankInvadersPopulation(scores, order, random);
        check(order.front() == 4, "Tie-breaking must preserve the unique best agent");
        for (std::size_t i = 1; i < order.size(); ++i)
            check(scores[order[i - 1]] >= scores[order[i]],
                  "Tie-breaking must never promote worse fitness over better fitness");
    }
}

int main()
{
    try
    {
        replay();
        collisions();
        wavesAndLimits();
        interpreterAndAgent();
        functionCalls();
        plateauSelection();
        std::puts("PASS: simulation, agent execution, function calls, and plateau selection");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
