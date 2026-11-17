#include "CnnEs.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

std::vector<double> cnnCenteredRanks(const std::vector<long long>& scores)
{
    if (scores.size() < 2 || std::any_of(scores.begin(), scores.end(), [](long long s) { return s < 0; }))
        throw std::invalid_argument("ES ranks require at least two complete evaluations");
    std::vector<std::size_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return scores[a] < scores[b]; });
    std::vector<double> ranks(scores.size());
    for (std::size_t begin = 0; begin < order.size();)
    {
        std::size_t end = begin + 1;
        while (end < order.size() && scores[order[end]] == scores[order[begin]]) ++end;
        const double rank = (0.5 * (begin + end - 1)) / (scores.size() - 1) - 0.5;
        for (std::size_t i = begin; i < end; ++i) ranks[order[i]] = rank;
        begin = end;
    }
    return ranks;
}

CnnEsGradient cnnEsGradient(const std::vector<CnnGenome>& noise,
                           const std::vector<long long>& scores, float sigma)
{
    if (noise.empty() || scores.size() != 2 * noise.size() || !std::isfinite(sigma) || sigma <= 0)
        throw std::invalid_argument("ES gradient requires complete mirrored pairs and positive sigma");
    const auto ranks = cnnCenteredRanks(scores);
    CnnEsGradient gradient{};
    const double scale = 1.0 / (scores.size() * double(sigma));
    // Reduction order is independent of worker scheduling and inference batching.
    for (std::size_t pair = 0; pair < noise.size(); ++pair)
    {
        const double coefficient = (ranks[2 * pair] - ranks[2 * pair + 1]) * scale;
        for (int p = 0; p < CnnGenome::Parameters; ++p)
            gradient[p] += coefficient * noise[pair].weights[p];
    }
    return gradient;
}

CnnEsOptimizer::CnnEsOptimizer(int population, const CnnEsSettings& settings)
    : mSettings(settings)
{
    if (population < 2 || population % 2)
        throw std::invalid_argument("ES requires an even --population of at least 2");
    if (!std::isfinite(settings.sigma) || settings.sigma <= 0 || settings.sigma > 8 ||
        !std::isfinite(settings.learningRate) || settings.learningRate <= 0 || settings.learningRate > 1 ||
        !std::isfinite(settings.weightDecay) || settings.weightDecay < 0 || settings.weightDecay > 1)
        throw std::invalid_argument("ES requires 0 < sigma <= 8, 0 < learning rate <= 1, and 0 <= weight decay <= 1");
    mNoise.resize(population / 2);
}

void CnnEsOptimizer::perturb(const CnnGenome& center, std::mt19937& random,
                           std::vector<CnnGenome>& population)
{
    population.resize(2 * mNoise.size());
    std::normal_distribution<float> normal(0, 1);
    for (std::size_t pair = 0; pair < mNoise.size(); ++pair)
        for (int p = 0; p < CnnGenome::Parameters; ++p)
        {
            const float epsilon = normal(random);
            mNoise[pair].weights[p] = epsilon;
            const float delta = mSettings.sigma * epsilon;
            // Temporary candidates are not checkpointed. Do not clip either side:
            // clipping would destroy the mirrored Gaussian perturbation at bounds.
            population[2 * pair].weights[p] = center.weights[p] + delta;
            population[2 * pair + 1].weights[p] = center.weights[p] - delta;
        }
    mPending = true;
}

CnnEsStep CnnEsOptimizer::update(CnnGenome& center, const std::vector<long long>& scores)
{
    if (!mPending) throw std::logic_error("ES update requires a newly evaluated perturbation population");
    const auto gradient = cnnEsGradient(mNoise, scores, mSettings.sigma);
    ++mStep;
    const double correction1 = 1 - std::pow(0.9, double(mStep));
    const double correction2 = 1 - std::pow(0.999, double(mStep));
    CnnEsStep result;
    for (int p = 0; p < CnnGenome::Parameters; ++p)
    {
        result.gradientNorm += gradient[p] * gradient[p];
        // Adam ascent on rank-shaped returns, with an L2 penalty on all weights
        // and biases. This is coupled L2 regularization, not AdamW decay.
        const double g = gradient[p] - double(mSettings.weightDecay) * center.weights[p];
        mFirst[p] = 0.9 * mFirst[p] + 0.1 * g;
        mSecond[p] = 0.999 * mSecond[p] + 0.001 * g * g;
        const double delta = mSettings.learningRate * (mFirst[p] / correction1) /
                             (std::sqrt(mSecond[p] / correction2) + 1e-8);
        const float next = static_cast<float>(std::max(-8.0, std::min(8.0, center.weights[p] + delta)));
        const double applied = double(next) - center.weights[p];
        result.stepNorm += applied * applied;
        center.weights[p] = next;
    }
    result.gradientNorm = std::sqrt(result.gradientNorm);
    result.stepNorm = std::sqrt(result.stepNorm);
    mPending = false;
    return result;
}
