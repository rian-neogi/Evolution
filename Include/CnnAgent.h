#pragma once

#include "SpaceInvaders.h"
#include <array>
#include <random>
#include <memory>
#include <string>
#include <vector>

// Grayscale silhouettes of the playfield, max-pooled over native 8x8 pixel blocks.
// Rectangles are pooled directly: no window, full-size framebuffer, or RAM features.
struct InvadersVision
{
    enum { Pool = 8, Width = SpaceInvaders::Width / Pool,
           Height = SpaceInvaders::Height / Pool, Size = Width * Height };
    using Image = std::array<float, Size>;
    static void observe(const SpaceInvaders::World& world, Image& image);
};

class CnnGenome
{
public:
    enum { Kernel = 3, Stride = 2, Filters = 4, Hidden = 16,
           Outputs = SpaceInvaders::ActionCount,
           ConvWidth = (InvadersVision::Width - Kernel) / Stride + 1,
           ConvHeight = (InvadersVision::Height - Kernel) / Stride + 1,
           Features = ConvWidth * ConvHeight * Filters,
           ConvBias = Filters * Kernel * Kernel,
           DenseWeights = ConvBias + Filters,
           DenseBias = DenseWeights + Features * Hidden,
           OutputWeights = DenseBias + Hidden,
           OutputBias = OutputWeights + Hidden * Outputs,
           Parameters = OutputBias + Outputs };

    std::array<float, Parameters> weights{};
    void randomize(std::mt19937& random);
    // Sparse Gaussian perturbations of weights AND biases; no gradients.
    void mutate(std::mt19937& random, float rate, float sigma);
    void save(const std::string& path) const;
    void load(const std::string& path);
};

// Separate from simulation and mutation RNGs. Keep one sampler per active game,
// and reset it to the episode seed whenever that game is reset or replaced.
class CnnActionSampler
{
public:
    explicit CnnActionSampler(std::uint32_t seed = 123) : mRandom(seed) {}
    void reset(std::uint32_t seed) { mRandom.seed(seed); }
    // Sample softmax(finite logits) at temperature 1, without heap allocation.
    int sample(const std::array<float, CnnGenome::Outputs>& logits);
private:
    std::mt19937 mRandom;
};

// A batch contains independent networks, not multiple inputs sharing one genome.
// Application scratch storage is allocated once, and matmul primitives are reused.
class CnnBatch
{
public:
    using Output = std::array<float, CnnGenome::Outputs>;
    explicit CnnBatch(int capacity);
    ~CnnBatch();
    // Bind a new network when a slot is filled. Multi-game batches copy its weights;
    // a singleton references the genome, which must outlive subsequent inference.
    void bind(int slot, const CnnGenome& genome);
    void predict(const InvadersVision::Image* inputs, int count, Output* outputs);
    void actions(SpaceInvaders* const* games, CnnActionSampler* const* samplers,
                 int count, int* actions);
    static const char* backend();
private:
    int mCapacity;
    std::vector<InvadersVision::Image> mInputs;
    std::vector<Output> mOutputs;
    std::vector<float> mPatches, mConvolution, mHidden;
    std::vector<CnnGenome> mGenomes;
    const CnnGenome* mSingleGenome = nullptr;
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

class CnnAgent
{
public:
    explicit CnnAgent(const CnnGenome& genome, std::uint32_t seed = 123)
        : mGenome(genome), mBatch(1), mSampler(seed) {}
    void reset(std::uint32_t seed) { mSampler.reset(seed); }
    int action(SpaceInvaders& game);
    std::array<float, CnnGenome::Outputs> predict(const InvadersVision::Image& input) const;
private:
    const CnnGenome& mGenome;
    mutable CnnBatch mBatch;
    CnnActionSampler mSampler;
};

// Read only the checkpoint type; each loader performs complete validation.
bool isCnnCheckpoint(const std::string& path);
