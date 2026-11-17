#pragma once

#include "CnnAgent.h"
#include <cstdint>

struct CnnEsSettings
{
    float sigma = 0.02f;
    float learningRate = 0.01f;
    float weightDecay = 0.005f;
};

using CnnEsGradient = std::array<double, CnnGenome::Parameters>;

// Ascending ranks scaled to [-0.5, 0.5], with average ranks for tied returns.
// Negative scores denote incomplete simulator evaluations and are rejected.
std::vector<double> cnnCenteredRanks(const std::vector<long long>& scores);

// One unit Gaussian direction per pair; scores are ordered +epsilon, -epsilon.
// g = sum((rank_plus - rank_minus) * epsilon) / (population * sigma).
CnnEsGradient cnnEsGradient(const std::vector<CnnGenome>& noise,
                           const std::vector<long long>& scores, float sigma);

struct CnnEsStep
{
    double gradientNorm = 0;
    double stepNorm = 0;
};

// Optimizer state lasts for one training run. Checkpoints remain policy-only.
class CnnEsOptimizer
{
public:
    CnnEsOptimizer(int population, const CnnEsSettings& settings);
    void perturb(const CnnGenome& center, std::mt19937& random,
                 std::vector<CnnGenome>& population);
    CnnEsStep update(CnnGenome& center, const std::vector<long long>& scores);
private:
    CnnEsSettings mSettings;
    std::vector<CnnGenome> mNoise;
    CnnEsGradient mFirst{}, mSecond{};
    std::uint64_t mStep = 0;
    bool mPending = false;
};
