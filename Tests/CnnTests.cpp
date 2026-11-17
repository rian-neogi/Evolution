#include "CnnAgent.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

static void check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

static void vision()
{
    SpaceInvaders::World world;
    world.lives = 0;
    InvadersVision::Image pooled;
    InvadersVision::observe(world, pooled);
    check(std::all_of(pooled.begin(), pooled.end(), [](float x) { return x == 0; }), "Empty image is not black");

    // Compare direct pooling against actual full-resolution rasterization, covering
    // subpixel-cell intersections, overlap, and clipping at all four screen edges.
    const int positions[][2] = {{8, 8}, {0, 0}, {256, 240}, {-10, -10}, {128, 120}, {255, 1}};
    for (const auto& position : positions)
    {
        world.playerShot.active = true;
        world.playerShot.x = position[0];
        world.playerShot.y = position[1];
        world.bullets[0].active = true;
        world.bullets[0].x = position[0] + 1;
        world.bullets[0].y = position[1] + 2;
        std::array<float, SpaceInvaders::Width * SpaceInvaders::Height> pixels{};
        const auto raster = [&](int x, int y, int width, int height, float gray) {
            for (int py = y; py < y + height; ++py)
                for (int px = x; px < x + width; ++px)
                    if (px >= 0 && px < SpaceInvaders::Width && py >= 0 && py < SpaceInvaders::Height)
                    {
                        float& pixel = pixels[py * SpaceInvaders::Width + px];
                        pixel = std::max(pixel, gray);
                    }
        };
        raster(position[0] - 1, position[1] - 3, 2, 6, 1);
        raster(position[0], position[1], 2, 5, 0.5f);
        InvadersVision::observe(world, pooled);
        for (int y = 0; y < InvadersVision::Height; ++y)
            for (int x = 0; x < InvadersVision::Width; ++x)
            {
                float maximum = 0;
                for (int dy = 0; dy < 8; ++dy)
                    for (int dx = 0; dx < 8; ++dx)
                        maximum = std::max(maximum, pixels[(8 * y + dy) * SpaceInvaders::Width + 8 * x + dx]);
                check(pooled[y * InvadersVision::Width + x] == maximum, "Direct pooling differs from pixel max pooling");
            }
    }
    SpaceInvaders game;
    auto visible = game.world();
    InvadersVision::observe(visible, pooled);
    check(std::any_of(pooled.begin(), pooled.end(), [](float x) { return x > 0; }), "Playfield has no pixels");
    const auto original = pooled;
    visible.score = 9999;
    visible.frame += 50;
    visible.lives = 2;
    InvadersVision::observe(visible, pooled);
    check(pooled == original, "Vision must not leak score, time, or HUD values into the image");
    visible.playerX -= 16;
    InvadersVision::observe(visible, pooled);
    check(pooled != original, "Ship movement must change the visual input");
}

static void forward()
{
    check(CnnGenome::Parameters == 13598, "Unexpected parameter count");
    CnnGenome genome;
    CnnAgent agent(genome);
    InvadersVision::Image input;
    input.fill(1);
    auto output = agent.predict(input);
    for (float value : output) check(value == 0, "Zero network must have zero logits");

    // A fully connected all-ones network has a closed-form answer in every layer.
    genome.weights.fill(1);
    for (int i = 0; i < CnnGenome::Outputs; ++i) genome.weights[CnnGenome::OutputBias + i] = float(i);
    output = agent.predict(input);
    for (int i = 0; i < CnnGenome::Outputs; ++i)
        check(output[i] == 134416.0f + i, "Convolution/dense/output sums are wrong");

    // Route one off-center pixel through filter 2, dense neuron 7, and output 5.
    // This checks kernel position, stride, flattening order, biases, and action mapping.
    genome.weights.fill(0);
    input.fill(0);
    input[11 * InvadersVision::Width + 14] = 0.75f;
    const int kernel = 2 * 9 + 1 * 3 + 2;
    const int feature = (2 * CnnGenome::ConvHeight + 5) * CnnGenome::ConvWidth + 6;
    genome.weights[kernel] = 2;
    genome.weights[CnnGenome::DenseWeights + 7 * CnnGenome::Features + feature] = 3;
    genome.weights[CnnGenome::DenseBias + 7] = -1;
    genome.weights[CnnGenome::OutputWeights + 5 * CnnGenome::Hidden + 7] = 2;
    genome.weights[CnnGenome::OutputBias + 5] = 0.25f;
    output = agent.predict(input);
    check(output[5] == 7.25f, "Sparse visual signal did not reach the expected action");
    for (int i = 0; i < 5; ++i) check(output[i] == 0, "Signal leaked into an unrelated output");
    genome.weights[kernel] = -2;
    genome.weights[CnnGenome::DenseBias + 7] = 0;
    genome.weights[CnnGenome::OutputBias + 5] = 0;
    check(std::abs(agent.predict(input)[5] + 0.0009f) < 1e-7f, "Leaky ReLU is incorrect");
    genome.weights.fill(0);
    for (int i = 0; i < CnnGenome::Outputs; ++i) genome.weights[CnnGenome::OutputBias + i] = -8;
    genome.weights[CnnGenome::OutputBias + SpaceInvaders::RightFire] = 8;
    SpaceInvaders game;
    check(agent.action(game) == SpaceInvaders::RightFire, "CNN logits mapped to the wrong action");
    check(game.world().frame == 0, "Inference must not advance the simulation");
}

static void softmax()
{
    CnnBatch::Output logits;
    for (int i = 0; i < CnnGenome::Outputs; ++i) logits[i] = std::log(float(i + 1));
    CnnActionSampler sampler(123);
    std::array<int, CnnGenome::Outputs> counts{};
    const int draws = 60000;
    for (int i = 0; i < draws; ++i)
    {
        const int action = sampler.sample(logits);
        check(action >= 0 && action < CnnGenome::Outputs, "Softmax sampled an invalid action");
        ++counts[action];
    }
    // exp(log(i+1)) gives exact target probabilities (i+1)/21.
    for (int i = 0; i < CnnGenome::Outputs; ++i)
        check(std::abs(double(counts[i]) / draws - double(i + 1) / 21) < 0.008,
              "Action frequencies do not follow softmax probabilities");

    CnnActionSampler replay(123), shifted(123), other(456);
    sampler.reset(123);
    CnnBatch::Output offset;
    for (int i = 0; i < CnnGenome::Outputs; ++i) offset[i] = logits[i] + 100;
    bool differentSeed = false;
    for (int i = 0; i < 1000; ++i)
    {
        const int action = sampler.sample(logits);
        check(action == replay.sample(logits), "Reset did not replay the action stream");
        check(action == shifted.sample(offset), "Adding a constant changed softmax sampling");
        differentSeed |= action != other.sample(logits);
    }
    check(differentSeed, "Action sampling ignores the seed");

    // Extreme finite values must neither overflow exp() nor select underflowed mass.
    for (int winner = 0; winner < CnnGenome::Outputs; ++winner)
    {
        logits.fill(-std::numeric_limits<float>::max());
        logits[winner] = std::numeric_limits<float>::max();
        for (int i = 0; i < 100; ++i)
            check(sampler.sample(logits) == winner, "Extreme logits broke softmax stability");
    }
    logits.fill(std::numeric_limits<float>::max());
    counts.fill(0);
    for (int i = 0; i < draws; ++i) ++counts[sampler.sample(logits)];
    for (int count : counts)
        check(std::abs(double(count) / draws - 1.0 / CnnGenome::Outputs) < 0.008,
              "Equal logits must sample all actions uniformly");
}

// Independent double-precision reference: direct image convolution and simple
// sequential dense sums, with no im2col, batching, or library matrix calls.
static std::array<double, CnnGenome::Outputs> reference(const CnnGenome& genome, const InvadersVision::Image& input)
{
    const auto& w = genome.weights;
    std::array<double, CnnGenome::Features> features;
    const auto relu = [](double x) { return x >= 0 ? x : 0.01 * x; };
    int feature = 0;
    for (int f = 0; f < CnnGenome::Filters; ++f)
        for (int y = 0; y < CnnGenome::ConvHeight; ++y)
            for (int x = 0; x < CnnGenome::ConvWidth; ++x)
            {
                double sum = w[CnnGenome::ConvBias + f];
                for (int dy = 0; dy < 3; ++dy)
                    for (int dx = 0; dx < 3; ++dx)
                        sum += double(w[f * 9 + dy * 3 + dx]) * input[(2 * y + dy) * InvadersVision::Width + 2 * x + dx];
                features[feature++] = relu(sum);
            }
    std::array<double, CnnGenome::Hidden> hidden;
    for (int n = 0; n < CnnGenome::Hidden; ++n)
    {
        double sum = w[CnnGenome::DenseBias + n];
        for (int f = 0; f < CnnGenome::Features; ++f)
            sum += double(w[CnnGenome::DenseWeights + n * CnnGenome::Features + f]) * features[f];
        hidden[n] = relu(sum);
    }
    std::array<double, CnnGenome::Outputs> output;
    for (int a = 0; a < CnnGenome::Outputs; ++a)
    {
        double sum = w[CnnGenome::OutputBias + a];
        for (int n = 0; n < CnnGenome::Hidden; ++n)
            sum += double(w[CnnGenome::OutputWeights + a * CnnGenome::Hidden + n]) * hidden[n];
        output[a] = sum;
    }
    return output;
}

static void batching()
{
    std::mt19937 random(917);
    std::uniform_real_distribution<float> pixel(-1, 1);
    const int capacity = 7;
    std::array<CnnGenome, capacity> genomes;
    std::array<InvadersVision::Image, capacity> inputs;
    std::array<CnnBatch::Output, capacity> outputs;
    CnnBatch batch(capacity);
    for (int pass = 0; pass < 3; ++pass)
    {
        for (int i = 0; i < capacity; ++i)
        {
            genomes[i].randomize(random);
            genomes[i].mutate(random, 1, 0.1f);
            for (float& value : inputs[i]) value = pixel(random);
            batch.bind(i, genomes[i]);
        }
        for (int count : {7, 1, 3, 0, 6, 7})
        {
            for (auto& row : outputs) row.fill(12345);
            batch.predict(inputs.data(), count, outputs.data());
            for (int i = 0; i < capacity; ++i)
            {
                if (i >= count)
                {
                    for (float value : outputs[i]) check(value == 12345, "Batch wrote an inactive output");
                    continue;
                }
                const auto expected = reference(genomes[i], inputs[i]);
                CnnAgent singleton(genomes[i]);
                check(outputs[i] == singleton.predict(inputs[i]), "Batch composition changed logits versus single-game inference");
                for (int a = 0; a < CnnGenome::Outputs; ++a)
                    check(std::abs(outputs[i][a] - expected[a]) < 2e-5 * (1 + std::abs(expected[a])),
                          "Independent network in batch disagrees with reference math");
            }
        }
    }
    // Use distinct constant policies to catch mixing up game/network slots.
    std::array<SpaceInvaders, capacity> games;
    std::array<SpaceInvaders*, capacity> pointers;
    std::array<CnnActionSampler, capacity> samplers;
    std::array<CnnActionSampler*, capacity> samplerPointers;
    std::array<int, capacity> actions;
    for (int i = 0; i < capacity; ++i)
    {
        genomes[i].weights.fill(0);
        for (int a = 0; a < CnnGenome::Outputs; ++a) genomes[i].weights[CnnGenome::OutputBias + a] = -8;
        genomes[i].weights[CnnGenome::OutputBias + i % CnnGenome::Outputs] = 8;
        batch.bind(i, genomes[i]);
        pointers[i] = &games[i];
        samplerPointers[i] = &samplers[i];
    }
    batch.actions(pointers.data(), samplerPointers.data(), capacity, actions.data());
    for (int i = 0; i < capacity; ++i)
    {
        check(actions[i] == i % CnnGenome::Outputs, "Batch mixed different agents' actions");
        check(games[i].world().frame == 0, "Batched inference advanced a game");
    }
    // Move games between batch rows, temporarily omit some, and compare their
    // stochastic action streams with independently evaluated singleton agents.
    std::array<std::unique_ptr<CnnAgent>, capacity> singletons;
    for (int i = 0; i < capacity; ++i)
    {
        for (int a = 0; a < CnnGenome::Outputs; ++a)
            genomes[i].weights[CnnGenome::OutputBias + a] = float((a + i) % CnnGenome::Outputs) - 3;
        samplers[i].reset(42 + i);
        singletons[i].reset(new CnnAgent(genomes[i], 42 + i));
    }
    int rotation = 0;
    for (int count : {7, 3, 0, 6, 1, 7})
    {
        for (int i = 0; i < count; ++i)
        {
            const int slot = (i + rotation) % capacity;
            batch.bind(i, genomes[slot]);
            pointers[i] = &games[slot];
            samplerPointers[i] = &samplers[slot];
        }
        batch.actions(pointers.data(), samplerPointers.data(), count, actions.data());
        for (int i = 0; i < count; ++i)
        {
            const int slot = (i + rotation) % capacity;
            check(actions[i] == singletons[slot]->action(games[slot]),
                  "Batch compaction changed a game's stochastic action stream");
        }
        ++rotation;
    }
    for (int bad : {-1, capacity + 1})
    {
        bool rejected = false;
        try { batch.predict(inputs.data(), bad, outputs.data()); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "Invalid batch size was accepted");
    }
    // Zero-sized inference must also accept absent buffers.
    batch.predict(nullptr, 0, nullptr);
    batch.actions(nullptr, nullptr, 0, nullptr);
}

static void mutation()
{
    CnnGenome parent, replay;
    std::mt19937 random(123), other(123);
    parent.randomize(random);
    replay.randomize(other);
    check(parent.weights == replay.weights, "Initialization must replay from a seed");
    auto child = parent;
    child.mutate(random, 0.1f, 0.05f);
    replay.mutate(other, 0.1f, 0.05f);
    check(child.weights == replay.weights, "Mutation must replay from a seed");
    int changed = 0;
    for (int i = 0; i < CnnGenome::Parameters; ++i) changed += child.weights[i] != parent.weights[i];
    check(changed > 0 && changed < CnnGenome::Parameters, "Sparse mutation must alter some parameters");
    child = parent;
    child.mutate(random, 1.0f, 0.05f);
    for (int index : {0, int(CnnGenome::ConvBias), int(CnnGenome::DenseWeights), int(CnnGenome::DenseBias),
                      int(CnnGenome::OutputWeights), int(CnnGenome::OutputBias)})
        check(child.weights[index] != parent.weights[index], "All layers and biases must be evolvable");
    child.weights.fill(8);
    child.mutate(random, 1, 8);
    for (float weight : child.weights)
        check(std::isfinite(weight) && std::abs(weight) <= 8, "Mutation exceeded finite weight limits");
    bool rejected = false;
    try { child.mutate(random, 0, 1); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid mutation settings must be rejected");
}

int main()
{
    try
    {
        vision();
        forward();
        softmax();
        mutation();
        batching();
        std::puts("PASS: CNN pixel pooling, forward math, softmax sampling, independent batched networks, and evolutionary mutation");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
