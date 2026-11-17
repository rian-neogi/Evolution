#include "CnnAgent.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>
#ifdef EVOLUTION_HAS_DNNL
#include "DnnlApi.h"
#include <omp.h>
#endif

namespace {
// Max compositing followed by max pooling can be applied directly to rectangles.
// Even a one-pixel overlap occupies its pooled cell, retaining small projectiles.
void rectangle(InvadersVision::Image& image, int x, int y, int width, int height, float gray)
{
    const int left = std::max(0, x), top = std::max(0, y);
    const int right = std::min(int(SpaceInvaders::Width), x + width);
    const int bottom = std::min(int(SpaceInvaders::Height), y + height);
    if (left >= right || top >= bottom) return;
    for (int row = top / InvadersVision::Pool; row <= (bottom - 1) / InvadersVision::Pool; ++row)
        for (int col = left / InvadersVision::Pool; col <= (right - 1) / InvadersVision::Pool; ++col)
        {
            float& pixel = image[row * InvadersVision::Width + col];
            pixel = std::max(pixel, gray);
        }
}

float activate(float value) { return value >= 0 ? value : 0.01f * value; }

bool validWeight(float value) { return std::isfinite(value) && std::abs(value) <= 8.0f; }

#ifndef EVOLUTION_HAS_DNNL
// Portable fallback when oneDNN is unavailable or explicitly disabled.
float dot(const float* a, const float* b, int size)
{
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 3 < size; i += 4)
    {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float sum = (s0 + s1) + (s2 + s3);
    for (; i < size; ++i) sum += a[i] * b[i];
    return sum;
}
#endif
}

void InvadersVision::observe(const SpaceInvaders::World& world, Image& image)
{
    image.fill(0);
    for (int row = 0; row < SpaceInvaders::Rows; ++row)
        for (int col = 0; col < SpaceInvaders::Columns; ++col)
            if (world.invaders[row] & (1u << col))
            {
                const int x = SpaceInvaders::enemyX(world, col), y = SpaceInvaders::enemyY(world, row);
                rectangle(image, x - 5, y - 3, 10, 6, 0.75f);
                rectangle(image, x - 3, y - 5, 6, 2, 0.75f);
                rectangle(image, x - 6, y + 3, 2, 3, 0.75f);
                rectangle(image, x + 4, y + 3, 2, 3, 0.75f);
            }
    for (int i = 0; i < SpaceInvaders::ShieldCells; ++i)
        if (world.shields[i])
            rectangle(image, SpaceInvaders::shieldX(i), SpaceInvaders::shieldY(i),
                      4, 4, 0.25f * world.shields[i]);
    if (world.lives)
    {
        rectangle(image, world.playerX - 7, SpaceInvaders::PlayerY - 2, 14, 6, 1.0f);
        rectangle(image, world.playerX - 2, SpaceInvaders::PlayerY - 6, 4, 4, 1.0f);
    }
    if (world.playerShot.active)
        rectangle(image, world.playerShot.x - 1, world.playerShot.y - 3, 2, 6, 1.0f);
    for (const auto& bullet : world.bullets)
        if (bullet.active) rectangle(image, bullet.x - 1, bullet.y - 2, 2, 5, 0.5f);
    if (world.saucerActive)
    {
        rectangle(image, world.saucerX - 9, 16, 18, 4, 0.75f);
        rectangle(image, world.saucerX - 5, 13, 10, 3, 0.75f);
    }
}

void CnnGenome::randomize(std::mt19937& random)
{
    weights.fill(0);
    const auto layer = [&](int begin, int end, int fanIn) {
        const float limit = std::sqrt(6.0f / fanIn);
        std::uniform_real_distribution<float> distribution(-limit, limit);
        for (int i = begin; i < end; ++i) weights[i] = distribution(random);
    };
    layer(0, ConvBias, Kernel * Kernel);
    layer(DenseWeights, DenseBias, Features);
    layer(OutputWeights, OutputBias, Hidden);
}

void CnnGenome::mutate(std::mt19937& random, float rate, float sigma)
{
    if (!std::isfinite(rate) || rate <= 0 || rate > 1 ||
        !std::isfinite(sigma) || sigma <= 0 || sigma > 8)
        throw std::invalid_argument("CNN mutation requires 0 < rate <= 1 and 0 < sigma <= 8");
    std::bernoulli_distribution choose(rate);
    std::normal_distribution<float> noise(0, sigma);
    const auto perturb = [&](float& weight) {
        weight = std::max(-8.0f, std::min(8.0f, weight + noise(random)));
    };
    bool mutated = false;
    for (auto& weight : weights)
        if (choose(random)) { perturb(weight); mutated = true; }
    if (!mutated) //always mutate atleast one weight
    {
        std::uniform_int_distribution<int> index(0, Parameters - 1);
        perturb(weights[index(random)]);
    }
}

#ifndef EVOLUTION_HAS_DNNL
static CnnBatch::Output predictScalar(const CnnGenome& genome, const InvadersVision::Image& input)
{
    const auto& weights = genome.weights;
    std::array<float, CnnGenome::Features> convolution;
    for (int filter = 0; filter < CnnGenome::Filters; ++filter)
        for (int y = 0; y < CnnGenome::ConvHeight; ++y)
            for (int x = 0; x < CnnGenome::ConvWidth; ++x)
            {
                float value = weights[CnnGenome::ConvBias + filter];
                for (int ky = 0; ky < CnnGenome::Kernel; ++ky)
                    for (int kx = 0; kx < CnnGenome::Kernel; ++kx)
                        value += input[(y * CnnGenome::Stride + ky) * InvadersVision::Width +
                                       x * CnnGenome::Stride + kx] *
                                 weights[(filter * CnnGenome::Kernel + ky) * CnnGenome::Kernel + kx];
                convolution[(filter * CnnGenome::ConvHeight + y) * CnnGenome::ConvWidth + x] = activate(value);
            }
    std::array<float, CnnGenome::Hidden> hidden;
    for (int neuron = 0; neuron < CnnGenome::Hidden; ++neuron)
    {
        const float* row = weights.data() + CnnGenome::DenseWeights + neuron * CnnGenome::Features;
        const float value = weights[CnnGenome::DenseBias + neuron] + dot(row, convolution.data(), CnnGenome::Features);
        hidden[neuron] = activate(value);
    }
    std::array<float, CnnGenome::Outputs> output;
    for (int action = 0; action < CnnGenome::Outputs; ++action)
    {
        const float* row = weights.data() + CnnGenome::OutputWeights + action * CnnGenome::Hidden;
        output[action] = weights[CnnGenome::OutputBias + action] + dot(row, hidden.data(), CnnGenome::Hidden);
    }
    return output;
}

#endif

namespace {
constexpr int Spatial = CnnGenome::ConvWidth * CnnGenome::ConvHeight;
constexpr int KernelArea = CnnGenome::Kernel * CnnGenome::Kernel;

int checkedCapacity(int capacity)
{
    if (capacity < 1 || capacity > 256) throw std::invalid_argument("CNN batch capacity must be in [1, 256]");
    return capacity;
}

}

#ifdef EVOLUTION_HAS_DNNL
namespace {
void checkDnnl(dnnl_status_t status)
{
    if (status != dnnl_success)
        throw std::runtime_error("oneDNN CPU matmul failed (status " + std::to_string(status) + ")");
}

template<class T, dnnl_status_t (*Destroy)(T)>
struct DnnlHandle
{
    T value = nullptr;
    DnnlHandle() = default;
    DnnlHandle(const DnnlHandle&) = delete;
    DnnlHandle& operator=(const DnnlHandle&) = delete;
    ~DnnlHandle() { if (value) Destroy(value); }
};

struct BatchMatmul
{
    DnnlHandle<dnnl_memory_t, dnnl_memory_destroy> a, b, c;
    DnnlHandle<dnnl_primitive_t, dnnl_primitive_destroy> primitive;
    void initialize(dnnl_engine_t engine, int batch, int rows, int columns, int inner,
                    const std::array<dnnl_dim_t, 3>& aStrides,
                    const std::array<dnnl_dim_t, 3>& bStrides, float* aData, float* bData, float* cData)
    {
        DnnlHandle<dnnl_memory_desc_t, dnnl_memory_desc_destroy> ad, bd, cd;
        const dnnl_dim_t aShape[] = {batch, rows, inner}, bShape[] = {batch, inner, columns};
        const dnnl_dim_t cShape[] = {batch, rows, columns}, cStrides[] = {rows * columns, columns, 1};
        checkDnnl(dnnl_memory_desc_create_with_strides(&ad.value, 3, aShape, dnnl_f32, aStrides.data()));
        checkDnnl(dnnl_memory_desc_create_with_strides(&bd.value, 3, bShape, dnnl_f32, bStrides.data()));
        checkDnnl(dnnl_memory_desc_create_with_strides(&cd.value, 3, cShape, dnnl_f32, cStrides));
        DnnlHandle<dnnl_primitive_desc_t, dnnl_primitive_desc_destroy> descriptor;
        checkDnnl(dnnl_matmul_primitive_desc_create(&descriptor.value, engine, ad.value, bd.value,
                                                  nullptr, cd.value, nullptr));
        checkDnnl(dnnl_primitive_create(&primitive.value, descriptor.value));
        checkDnnl(dnnl_memory_create(&a.value, ad.value, engine, aData));
        checkDnnl(dnnl_memory_create(&b.value, bd.value, engine, bData));
        checkDnnl(dnnl_memory_create(&c.value, cd.value, engine, cData));
    }
    void execute(dnnl_stream_t stream)
    {
        // Public oneDNN execution argument IDs: SRC=1, WEIGHTS=33, DST=17.
        const dnnl_exec_arg_t arguments[] = {{1, a.value}, {33, b.value}, {17, c.value}};
        checkDnnl(dnnl_primitive_execute(primitive.value, stream, 3, arguments));
        checkDnnl(dnnl_stream_wait(stream));
    }
};
}

struct CnnBatch::Impl
{
    DnnlHandle<dnnl_engine_t, dnnl_engine_destroy> engine;
    DnnlHandle<dnnl_stream_t, dnnl_stream_destroy> stream;
    BatchMatmul conv, dense, output;
    Impl(int batch, float* weights, float* patches, float* convolution, float* hidden, float* logits)
    {
        const auto* version = dnnl_version();
        // Public runtime values: SEQ=0, OMP=2. Other runtimes cannot be capped by OpenMP.
        if (version->major != 3 || (version->cpu_runtime != 0 && version->cpu_runtime != 2))
            throw std::runtime_error("CNN requires oneDNN v3 with the OpenMP or sequential CPU runtime");
        // OpenMP settings are thread-local: each training worker constructs its own batch.
        omp_set_num_threads(1);
        checkDnnl(dnnl_engine_create(&engine.value, dnnl_cpu, 0));
        checkDnnl(dnnl_stream_create(&stream.value, engine.value, 1));
        conv.initialize(engine.value, batch, CnnGenome::Filters, Spatial, KernelArea,
                        {{CnnGenome::Parameters, KernelArea, 1}}, {{KernelArea * Spatial, Spatial, 1}},
                        weights, patches, convolution);
        dense.initialize(engine.value, batch, 1, CnnGenome::Hidden, CnnGenome::Features,
                         {{CnnGenome::Features, CnnGenome::Features, 1}},
                         {{CnnGenome::Parameters, 1, CnnGenome::Features}},
                         convolution, weights + CnnGenome::DenseWeights, hidden);
        output.initialize(engine.value, batch, 1, CnnGenome::Outputs, CnnGenome::Hidden,
                          {{CnnGenome::Hidden, CnnGenome::Hidden, 1}},
                          {{CnnGenome::Parameters, 1, CnnGenome::Hidden}},
                          hidden, weights + CnnGenome::OutputWeights, logits);
    }
    void bindSingle(const CnnGenome& genome)
    {
        // oneDNN's input handles are void*, but inference never writes these weights.
        float* weights = const_cast<float*>(genome.weights.data());
        checkDnnl(dnnl_memory_set_data_handle(conv.a.value, weights));
        checkDnnl(dnnl_memory_set_data_handle(dense.b.value, weights + CnnGenome::DenseWeights));
        checkDnnl(dnnl_memory_set_data_handle(output.b.value, weights + CnnGenome::OutputWeights));
    }
};
#else
struct CnnBatch::Impl {};
#endif

CnnBatch::CnnBatch(int capacity)
    : mCapacity(checkedCapacity(capacity)), mInputs(mCapacity), mOutputs(mCapacity),
      mPatches(mCapacity * KernelArea * Spatial), mConvolution(mCapacity * CnnGenome::Features),
      mHidden(mCapacity * CnnGenome::Hidden), mGenomes(mCapacity)
{
#ifdef EVOLUTION_HAS_DNNL
    static_assert(sizeof(CnnGenome) == CnnGenome::Parameters * sizeof(float), "CNN weights must be contiguous");
    mImpl.reset(new Impl(mCapacity, mGenomes[0].weights.data(), mPatches.data(),
                        mConvolution.data(), mHidden.data(), mOutputs[0].data()));
#endif
}

CnnBatch::~CnnBatch() = default;

const char* CnnBatch::backend()
{
#ifdef EVOLUTION_HAS_DNNL
    return "oneDNN batched matmul (1 math thread per worker)";
#else
    return "portable scalar (oneDNN disabled/unavailable)";
#endif
}

void CnnBatch::bind(int slot, const CnnGenome& genome)
{
    if (slot < 0 || slot >= mCapacity) throw std::invalid_argument("CNN slot exceeds its batch capacity");
    if (mCapacity == 1)
    {
        if (mSingleGenome == &genome) return;
#ifdef EVOLUTION_HAS_DNNL
        mImpl->bindSingle(genome);
#endif
        mSingleGenome = &genome;
    }
    else mGenomes[slot] = genome;
}

void CnnBatch::predict(const InvadersVision::Image* inputs, int count, Output* outputs)
{
    if (count < 0 || count > mCapacity) throw std::invalid_argument("CNN batch exceeds its capacity");
    if (count == 0) return;
#ifdef EVOLUTION_HAS_DNNL
    // im2col: [batch, 9, 210]. The convolution matmul [batch, 4, 9] times
    // [batch, 9, 210] preserves the checkpoint's filter-major feature order.
    for (int i = 0; i < count; ++i)
    {
        float* patches = mPatches.data() + i * KernelArea * Spatial;
        for (int ky = 0; ky < CnnGenome::Kernel; ++ky)
            for (int kx = 0; kx < CnnGenome::Kernel; ++kx)
                for (int y = 0; y < CnnGenome::ConvHeight; ++y)
                    for (int x = 0; x < CnnGenome::ConvWidth; ++x)
                        patches[(ky * CnnGenome::Kernel + kx) * Spatial + y * CnnGenome::ConvWidth + x] =
                            inputs[i][(y * CnnGenome::Stride + ky) * InvadersVision::Width + x * CnnGenome::Stride + kx];
    }
    // Fixed capacity reuses JIT primitives even as the final batch drains.
    // Inactive rows are independent and ignored; no simulator advances for them.
    mImpl->conv.execute(mImpl->stream.value);
    for (int i = 0; i < count; ++i)
    {
        const auto& weights = (mSingleGenome ? *mSingleGenome : mGenomes[i]).weights;
        float* features = mConvolution.data() + i * CnnGenome::Features;
        for (int f = 0; f < CnnGenome::Filters; ++f)
            for (int p = 0; p < Spatial; ++p)
                features[f * Spatial + p] = activate(features[f * Spatial + p] + weights[CnnGenome::ConvBias + f]);
    }
    mImpl->dense.execute(mImpl->stream.value);
    for (int i = 0; i < count; ++i)
    {
        const auto& weights = (mSingleGenome ? *mSingleGenome : mGenomes[i]).weights;
        float* hidden = mHidden.data() + i * CnnGenome::Hidden;
        for (int n = 0; n < CnnGenome::Hidden; ++n)
            hidden[n] = activate(hidden[n] + weights[CnnGenome::DenseBias + n]);
    }
    mImpl->output.execute(mImpl->stream.value);
    for (int i = 0; i < count; ++i)
        for (int a = 0; a < CnnGenome::Outputs; ++a)
            outputs[i][a] = mOutputs[i][a] + (mSingleGenome ? *mSingleGenome : mGenomes[i]).weights[CnnGenome::OutputBias + a];
#else
    for (int i = 0; i < count; ++i)
        outputs[i] = predictScalar(mSingleGenome ? *mSingleGenome : mGenomes[i], inputs[i]);
#endif
}

int CnnActionSampler::sample(const std::array<float, CnnGenome::Outputs>& logits)
{
    // Subtracting the maximum keeps exp() in [0, 1], even for large evolved logits.
    const double maximum = *std::max_element(logits.begin(), logits.end());
    std::array<double, CnnGenome::Outputs> weights;
    double total = 0;
    for (int i = 0; i < CnnGenome::Outputs; ++i)
    {
        weights[i] = std::exp(static_cast<double>(logits[i]) - maximum);
        total += weights[i];
    }
    // Drawing against the unnormalized sum is equivalent to dividing by total.
    double remaining = std::uniform_real_distribution<double>(0, total)(mRandom);
    int lastPositive = 0;
    for (int i = 0; i < CnnGenome::Outputs; ++i)
        if (weights[i] > 0)
        {
            lastPositive = i;
            if (remaining < weights[i]) return i;
            remaining -= weights[i];
        }
    return lastPositive; // Rounding at the upper endpoint must not select zero mass.
}

void CnnBatch::actions(SpaceInvaders* const* games, CnnActionSampler* const* samplers,
                       int count, int* actions)
{
    if (count < 0 || count > mCapacity) throw std::invalid_argument("CNN batch exceeds its capacity");
    for (int i = 0; i < count; ++i) InvadersVision::observe(games[i]->world(), mInputs[i]);
    predict(mInputs.data(), count, mOutputs.data());
    for (int i = 0; i < count; ++i)
        actions[i] = samplers[i]->sample(mOutputs[i]);
}

std::array<float, CnnGenome::Outputs> CnnAgent::predict(const InvadersVision::Image& input) const
{
    mBatch.bind(0, mGenome);
    CnnBatch::Output output;
    mBatch.predict(&input, 1, &output);
    return output;
}

int CnnAgent::action(SpaceInvaders& game)
{
    mBatch.bind(0, mGenome);
    SpaceInvaders* world = &game;
    int action;
    CnnActionSampler* sampler = &mSampler;
    mBatch.actions(&world, &sampler, 1, &action);
    return action;
}

void CnnGenome::save(const std::string& path) const
{
    for (float weight : weights)
        if (!validWeight(weight)) throw std::runtime_error("Invalid CNN weight; checkpoint not saved");
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Cannot open checkpoint for writing: " + path);
    file.imbue(std::locale::classic());
    file << "EVOLUTION_CNN 1\n" << InvadersVision::Width << ' ' << InvadersVision::Height << ' '
         << InvadersVision::Pool << ' ' << Kernel << ' ' << Stride << ' ' << Filters << ' '
         << Hidden << ' ' << Outputs << ' ' << Parameters << '\n';
    file << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (float weight : weights) file << weight << '\n';
    file.close();
    if (!file) throw std::runtime_error("Failed to write checkpoint: " + path);
}

void CnnGenome::load(const std::string& path)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open checkpoint: " + path);
    file.imbue(std::locale::classic());
    const auto invalid = [&] { return std::runtime_error("Invalid or incompatible CNN checkpoint: " + path); };
    std::string magic;
    int version;
    if (!(file >> magic >> version) || magic != "EVOLUTION_CNN" || version != 1) throw invalid();
    const int expected[] = {InvadersVision::Width, InvadersVision::Height, InvadersVision::Pool,
                           Kernel, Stride, Filters, Hidden, Outputs, Parameters};
    for (int value : expected)
    {
        int actual;
        if (!(file >> actual) || actual != value) throw invalid();
    }
    std::array<float, Parameters> loaded;
    for (float& weight : loaded)
        if (!(file >> weight) || !validWeight(weight)) throw invalid();
    file >> std::ws;
    if (!file.eof()) throw invalid();
    weights = loaded;
}

bool isCnnCheckpoint(const std::string& path)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open checkpoint: " + path);
    std::string magic;
    file >> magic;
    return magic == "EVOLUTION_CNN";
}
