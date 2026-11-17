#pragma once
#include "SpaceInvaders.h"
#include "CnnEs.h"
#include <random>
#include <string>
#include <vector>

enum class AgentType { Program, Cnn };
enum class TrainingOptimizer { Genetic, Es };

struct RunOptions
{
    AgentType agent = AgentType::Program;
    TrainingOptimizer optimizer = TrainingOptimizer::Genetic;
    CnnEsSettings es;
    float mutationRate = 0.1f;
    float mutationSigma = 0.05f;
    std::uint32_t seed = 123;
    int maxFrames = 3600;
    int episodes = 3;
    int population = 128;
    int elite = 8;
    int generations = 100;
    int threads = 1;
    int batchSize = 4;
    int fps = 60;
    int benchmarkFrames = 1000000;
    bool fresh = false;
    bool indefinite = false;
    std::string loadPath;
    std::string savePath = "invaders_best";
};

// Owns a fixed register bank; each action uses the same interpreter as legacy training.
class InvadersAgent
{
public:
    explicit InvadersAgent(const Genotype& genotype);
    int action(SpaceInvaders& game);
    InvadersAgent(const InvadersAgent&) = delete;
    InvadersAgent& operator=(const InvadersAgent&) = delete;
private:
    const Genotype& mGenotype;
    std::array<int, 8> mRegisters{};
    Execution mExecution;
};

// Rank by descending fitness, using seeded randomness to break ties fairly.
void rankInvadersPopulation(const std::vector<long long>& scores, std::vector<int>& order,
                           std::mt19937& tieRandom);
void trainInvaders(const RunOptions& options);
void benchmarkInvaders(const RunOptions& options);
int viewInvaders(const RunOptions& options, bool human);
