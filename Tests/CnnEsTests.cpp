#include "CnnEs.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

static void check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template<class Function>
static void rejects(Function function)
{
    bool threw = false;
    try { function(); } catch (const std::exception&) { threw = true; }
    check(threw, "Invalid ES input was accepted");
}

static void ranksAndGradient()
{
    const auto ranks = cnnCenteredRanks({30, 10, 20, 20});
    check(ranks[0] == 0.5 && ranks[1] == -0.5 && ranks[2] == 0 && ranks[3] == 0,
          "Centered ranks or tie averaging are incorrect");
    for (double r : cnnCenteredRanks({7, 7, 7, 7})) check(r == 0, "Ties introduced a false ranking");
    rejects([] { cnnCenteredRanks({}); });
    rejects([] { cnnCenteredRanks({1}); });
    rejects([] { cnnCenteredRanks({1, -1}); });

    std::vector<CnnGenome> noise(2);
    noise[0].weights[0] = 2; noise[0].weights[1] = -1;
    noise[1].weights[0] = 1; noise[1].weights[1] = 3;
    const auto gradient = cnnEsGradient(noise, {40, 10, 20, 30}, 0.2f);
    check(std::abs(gradient[0] - 25.0 / 12) < 1e-6 && std::abs(gradient[1] + 2.5) < 1e-6,
          "Mirrored gradient has wrong sign, pair count, or sigma normalization");
    const auto reversed = cnnEsGradient(noise, {10, 40, 30, 20}, 0.2f);
    const auto scaledRewards = cnnEsGradient(noise, {4100, 1100, 2100, 3100}, 0.2f);
    const auto largerSigma = cnnEsGradient(noise, {40, 10, 20, 30}, 0.4f);
    const auto tied = cnnEsGradient(noise, {5, 5, 9, 9}, 0.2f);
    for (int p = 0; p < CnnGenome::Parameters; ++p)
    {
        check(reversed[p] == -gradient[p], "Swapping pair scores must reverse the gradient");
        check(scaledRewards[p] == gradient[p], "Rank shaping depends on reward scale or offset");
        check(largerSigma[p] == 0.5 * gradient[p], "Gradient does not scale inversely with sigma");
        check(tied[p] == 0, "Equal mirrored returns must cancel");
        if (p > 1) check(gradient[p] == 0, "Gradient leaked into an unperturbed coordinate");
    }
    rejects([&] { cnnEsGradient(noise, {1, 2}, 0.2f); });
    rejects([&] { cnnEsGradient(noise, {1, 2, -1, 3}, 0.2f); });
    rejects([&] { cnnEsGradient(noise, {1, 2, 3, 4}, 0); });
}

static void mirroredAndAdam()
{
    CnnEsSettings settings;
    settings.sigma = 0.2f; settings.learningRate = 0.03f; settings.weightDecay = 0;
    CnnEsOptimizer optimizer(2, settings);
    CnnGenome center;
    std::vector<CnnGenome> population;
    std::mt19937 random(123);
    rejects([&] { optimizer.update(center, {2, 1}); });
    optimizer.perturb(center, random, population);
    check(population.size() == 2, "Wrong perturbation population size");
    for (int p = 0; p < CnnGenome::Parameters; ++p)
        check(population[0].weights[p] == -population[1].weights[p], "Perturbations are not antithetic");
    // Every parameter, including all layers' biases, participates in dense noise.
    const int layerStarts[] = {0, CnnGenome::ConvBias, CnnGenome::DenseWeights, CnnGenome::DenseBias,
                              CnnGenome::OutputWeights, CnnGenome::OutputBias};
    for (int p : layerStarts)
        check(population[0].weights[p] != 0, "ES omitted a layer or its biases");
    const auto original = center.weights;
    rejects([&] { optimizer.update(center, {2, -1}); });
    check(center.weights == original, "Incomplete evaluations changed the center");

    // Independent scalar Adam calculation over two steps, including bias correction
    // and a reversed return ordering on the second step to exercise momentum.
    double m[3]{}, v[3]{};
    const int coordinates[] = {0, CnnGenome::DenseBias, CnnGenome::OutputBias + 5};
    for (int step = 1; step <= 2; ++step)
    {
        if (step == 2) optimizer.perturb(center, random, population);
        double expected[3];
        for (int i = 0; i < 3; ++i)
        {
            const int p = coordinates[i];
            const double epsilon = (double(population[0].weights[p]) - population[1].weights[p]) / (2 * settings.sigma);
            const double g = (step == 1 ? 1 : -1) * epsilon / (2 * settings.sigma);
            m[i] = .9 * m[i] + .1 * g;
            v[i] = .999 * v[i] + .001 * g * g;
            expected[i] = center.weights[p] + settings.learningRate * (m[i] / (1 - std::pow(.9, step))) /
                          (std::sqrt(v[i] / (1 - std::pow(.999, step))) + 1e-8);
        }
        const auto result = optimizer.update(center, step == 1 ? std::vector<long long>{2, 1} : std::vector<long long>{1, 2});
        check(result.gradientNorm > 0 && result.stepNorm > 0, "No ES update was applied");
        for (int i = 0; i < 3; ++i)
            check(std::abs(center.weights[coordinates[i]] - expected[i]) < 2e-7,
                  "Adam ascent, persistent moments, or bias correction is incorrect");
        rejects([&] { optimizer.update(center, {2, 1}); });
    }
    CnnEsOptimizer flat(4, settings);
    flat.perturb(center, random, population);
    const auto before = center.weights;
    const auto flatStep = flat.update(center, {1, 1, 1, 1});
    check(center.weights == before && flatStep.gradientNorm == 0 && flatStep.stepNorm == 0,
          "Flat returns with zero decay and fresh Adam state must not move the center");

    settings.weightDecay = .1f;
    CnnEsOptimizer decay(2, settings);
    center.weights.fill(2);
    decay.perturb(center, random, population);
    const auto decayStep = decay.update(center, {1, 1});
    check(decayStep.gradientNorm == 0 && std::abs(center.weights[0] - 1.97) < 1e-6,
          "L2 decay must shrink positive weights even when rewards tie");

    settings.weightDecay = 0;
    CnnEsOptimizer bounded(2, settings);
    center.weights.fill(8);
    bounded.perturb(center, random, population);
    bool outside = false;
    for (int p = 0; p < CnnGenome::Parameters; ++p)
    {
        check(std::abs(population[0].weights[p] + population[1].weights[p] - 16) < 2e-6,
              "Clipping candidates broke symmetry at a checkpoint boundary");
        outside |= population[0].weights[p] > 8 || population[1].weights[p] > 8;
    }
    check(outside, "Temporary Gaussian candidates were clipped");
    bounded.update(center, {2, 1});
    for (float weight : center.weights) check(std::isfinite(weight) && std::abs(weight) <= 8, "Center exceeds checkpoint bounds");
}

static void optimization()
{
    CnnEsSettings settings;
    settings.sigma = .1f; settings.learningRate = .05f; settings.weightDecay = 0;
    CnnEsOptimizer optimizer(32, settings);
    CnnGenome center;
    std::mt19937 random(917);
    std::vector<CnnGenome> population;
    const auto distance = [](const CnnGenome& g) {
        return std::pow(g.weights[0] - 1.0, 2) + std::pow(g.weights[1] + .5, 2);
    };
    const double initial = distance(center);
    for (int i = 0; i < 40; ++i)
    {
        optimizer.perturb(center, random, population);
        std::vector<long long> scores;
        for (const auto& candidate : population)
            scores.push_back(std::llround(1000000 - 10000 * distance(candidate)));
        optimizer.update(center, scores);
    }
    check(distance(center) < initial * .05, "ES failed to optimize a known two-coordinate objective");
}

int main()
{
    ranksAndGradient();
    mirroredAndAdam();
    optimization();
    CnnEsSettings settings;
    rejects([&] { CnnEsOptimizer bad(3, settings); });
    settings.sigma = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { CnnEsOptimizer bad(2, settings); });
    std::cout << "PASS: centered ranks, mirrored ES gradient, Adam, bounds, and toy optimization\n";
}
