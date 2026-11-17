#include "LocalTraining.h"
#include "CnnAgent.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <sys/stat.h>
#include <vector>

namespace {
// Only lock-free atomic operations are safe to call from a signal handler.
static_assert(ATOMIC_BOOL_LOCK_FREE == 2, "Ctrl+C handling requires lock-free atomic bool");
std::atomic<bool> stopRequested{false};

void onInterrupt(int) { stopRequested.store(true, std::memory_order_relaxed); }

class InterruptHandler
{
public:
    explicit InterruptHandler(bool enabled) : mEnabled(enabled)
    {
        stopRequested.store(false, std::memory_order_relaxed);
        if (mEnabled)
        {
            mPrevious = std::signal(SIGINT, onInterrupt);
            if (mPrevious == SIG_ERR) throw std::runtime_error("Cannot install Ctrl+C handler");
        }
    }
    ~InterruptHandler() { if (mEnabled) std::signal(SIGINT, mPrevious); }
private:
    bool mEnabled;
    using Handler = void (*)(int);
    Handler mPrevious = SIG_DFL;
};

bool interrupted(const RunOptions& options)
{
    return options.indefinite && stopRequested.load(std::memory_order_relaxed);
}

// The generic program agent retains its serial-per-worker episode evaluator.
template<class Genome, class Agent>
void evaluateQueuedAgents(const RunOptions& options, const std::vector<Genome>& population,
                          std::vector<long long>& scores, std::atomic<int>& next, long long& frames)
{
    SpaceInvaders game(options.seed, options.maxFrames);
    for (;;)
    {
        if (interrupted(options)) break;
        const int i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= options.population) break;
        Agent agent(population[i]);
        long long score = 0;
        int episode = 0;
        for (; episode < options.episodes; ++episode)
        {
            if (interrupted(options)) break;
            game.reset(options.seed + static_cast<std::uint32_t>(episode));
            while (!game.is_over())
            {
                // Check periodically so long evaluations stop promptly without
                // adding a signal check to every simulation tick.
                if ((game.world().frame & 63) == 0 && interrupted(options)) break;
                score += game.do_action(agent.action(game));
            }
            frames += game.world().frame;
            if (!game.is_over()) break;
        }
        // Only compare agents evaluated on the complete seed sequence.
        if (episode == options.episodes) scores[i] = score;
    }
}

int cnnBatchSize(const RunOptions& options)
{
    // Keep enough batches for all requested workers, even for small populations.
    const int workers = std::min(options.threads, options.population);
    return std::min(options.batchSize, (options.population + workers - 1) / workers);
}

template<>
void evaluateQueuedAgents<CnnGenome, CnnAgent>(const RunOptions& options,
    const std::vector<CnnGenome>& population, std::vector<long long>& scores,
    std::atomic<int>& next, long long& frames)
{
    const int capacity = cnnBatchSize(options);
    CnnBatch batch(capacity);
    std::vector<SpaceInvaders> games;
    games.reserve(capacity);
    std::vector<int> indices(capacity), episodes(capacity, 0), slots(capacity), actions(capacity);
    std::vector<long long> totals(capacity, 0);
    std::vector<const CnnGenome*> genomes(capacity);
    std::vector<SpaceInvaders*> activeGames(capacity);
    std::vector<CnnActionSampler> samplers(capacity);
    std::vector<CnnActionSampler*> activeSamplers(capacity);
    // Claim the initial block in one operation, distributing full batches to workers.
    const int begin = next.fetch_add(capacity, std::memory_order_relaxed);
    for (int slot = 0; slot < capacity; ++slot)
    {
        indices[slot] = begin + slot;
        games.emplace_back(options.seed, options.maxFrames);
        samplers[slot].reset(options.seed);
    }
    unsigned int tick = 0;
    for (;;)
    {
        if ((tick++ & 63) == 0 && interrupted(options)) break;
        int count = 0;
        for (int slot = 0; slot < capacity; ++slot)
            if (indices[slot] < options.population)
            {
                slots[count] = slot;
                const CnnGenome* genome = &population[indices[slot]];
                if (genomes[count] != genome)
                {
                    batch.bind(count, *genome);
                    genomes[count] = genome;
                }
                activeGames[count] = &games[slot];
                activeSamplers[count++] = &samplers[slot];
            }
        if (count == 0) break;
        batch.actions(activeGames.data(), activeSamplers.data(), count, actions.data());
        for (int i = 0; i < count; ++i)
        {
            const int slot = slots[i];
            SpaceInvaders& game = games[slot];
            totals[slot] += game.do_action(actions[i]);
            if (game.is_over())
            {
                if (++episodes[slot] == options.episodes)
                {
                    // Publish only a complete seed sequence. In-flight slots are
                    // discarded on Ctrl+C, just like partial serial evaluations.
                    scores[indices[slot]] = totals[slot];
                    indices[slot] = next.fetch_add(1, std::memory_order_relaxed);
                    episodes[slot] = 0;
                    totals[slot] = 0;
                }
                if (indices[slot] < options.population)
                {
                    const auto seed = options.seed + static_cast<std::uint32_t>(episodes[slot]);
                    game.reset(seed);
                    samplers[slot].reset(seed);
                }
            }
        }
        frames += count;
    }
}

// Workers live for the whole run and claim agents as they become available.
// Selection/mutation may only touch the population after evaluate() returns.
template<class Genome, class Agent>
class PopulationEvaluator
{
public:
    PopulationEvaluator(const RunOptions& options, const std::vector<Genome>& population,
                        std::vector<long long>& scores, int workers)
        : mOptions(options), mPopulation(population), mScores(scores), mFrames(workers)
    {
        mThreads.reserve(workers - 1);
        try
        {
            // The calling thread is worker zero.
            for (int worker = 1; worker < workers; ++worker)
                mThreads.emplace_back(&PopulationEvaluator::workerLoop, this, worker);
        }
        catch (...)
        {
            stop();
            throw;
        }
    }

    ~PopulationEvaluator() { stop(); }

    long long evaluate()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mNext.store(0, std::memory_order_relaxed);
            mFailure = nullptr;
            mPending = mThreads.size();
            ++mGeneration;
        }
        mWorkAvailable.notify_all();
        evaluateAgents(0);
        std::unique_lock<std::mutex> lock(mMutex);
        mWorkDone.wait(lock, [this] { return mPending == 0; });
        if (mFailure) std::rethrow_exception(mFailure);
        return std::accumulate(mFrames.begin(), mFrames.end(), 0LL);
    }

private:
    const RunOptions& mOptions;
    const std::vector<Genome>& mPopulation;
    std::vector<long long>& mScores;
    std::vector<long long> mFrames;
    std::vector<std::thread> mThreads;
    std::atomic<int> mNext{0};
    std::mutex mMutex;
    std::condition_variable mWorkAvailable, mWorkDone;
    std::size_t mGeneration = 0, mPending = 0;
    bool mStopping = false;
    std::exception_ptr mFailure;

    void evaluateAgents(int worker)
    {
        long long frames = 0;
        try
        {
            evaluateQueuedAgents<Genome, Agent>(mOptions, mPopulation, mScores, mNext, frames);
        }
        catch (...)
        {
            // Let in-flight evaluations finish, then report failure on the main thread.
            mNext.store(mOptions.population, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(mMutex);
            if (!mFailure) mFailure = std::current_exception();
        }
        mFrames[worker] = frames;
    }

    void workerLoop(int worker)
    {
        std::size_t generation = 0;
        std::unique_lock<std::mutex> lock(mMutex);
        for (;;)
        {
            mWorkAvailable.wait(lock, [&] { return mStopping || mGeneration != generation; });
            if (mStopping) return;
            generation = mGeneration;
            lock.unlock();
            evaluateAgents(worker);
            lock.lock();
            if (--mPending == 0) mWorkDone.notify_one();
        }
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mStopping = true;
        }
        mWorkAvailable.notify_all();
        for (auto& thread : mThreads) thread.join();
    }
};
}

InvadersAgent::InvadersAgent(const Genotype& genotype)
    : mGenotype(genotype), mExecution(mRegisters.data(), mRegisters.size(), SpaceInvaders::StateSize) {}

int InvadersAgent::action(SpaceInvaders& game)
{
    mExecution.mInputs = game.getState();
    mExecution.reset();
    mExecution.run_code(&mGenotype, 0);
    return mRegisters[0] % SpaceInvaders::ActionCount;
}

void rankInvadersPopulation(const std::vector<long long>& scores, std::vector<int>& order,
                           std::mt19937& tieRandom)
{
    order.resize(scores.size());
    std::iota(order.begin(), order.end(), 0);
    // Randomize equal-fitness order before sorting, so neutral offspring can
    // become parents. The RNG stays on the main thread, independent of workers.
    std::shuffle(order.begin(), order.end(), tieRandom);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
}

static std::string trainingLoadPath(const RunOptions& options, bool& continuingSavedBest)
{
    std::string loadPath = options.loadPath;
    continuingSavedBest = false;
    if (loadPath.empty() && !options.fresh)
    {
        struct stat info;
        if (stat(options.savePath.c_str(), &info) == 0)
        {
            loadPath = options.savePath;
            continuingSavedBest = true;
        }
        else if (errno != ENOENT)
            throw std::runtime_error("Cannot inspect checkpoint " + options.savePath + ": " + std::strerror(errno));
    }
    return loadPath;
}

template<class Genome>
static void saveTrainingBest(const Genome& genome, const RunOptions& options, long long score)
{
    const std::string temporary = options.savePath + ".tmp";
    genome.save(temporary);
    if (std::rename(temporary.c_str(), options.savePath.c_str()) != 0)
        throw std::runtime_error("Cannot replace checkpoint: " + options.savePath);
    std::cout << "Saved new best: " << options.savePath << " (mean score "
              << double(score) / options.episodes << ")\n" << std::flush;
}

template<class Genome, class Agent, class Initialize, class Mutate>
static void trainPopulation(const RunOptions& options, Initialize initialize, Mutate mutate)
{
    bool continuingSavedBest = false;
    const std::string loadPath = trainingLoadPath(options, continuingSavedBest);
    srand(options.seed);
    std::vector<Genome> population(options.population);
    if (!loadPath.empty())
    {
        population[0].load(loadPath);
        for (int i = 1; i < options.population; ++i)
        {
            population[i] = population[0];
            mutate(population[i]);
        }
        std::cout << "Starting from checkpoint: " << loadPath << '\n';
    }
    else
    {
        for (auto& genome : population) initialize(genome);
        std::cout << "Starting a fresh population\n";
    }
    std::vector<long long> scores(options.population);
    std::vector<int> order(options.population);
    std::mt19937 tieRandom(options.seed);
    const int workers = std::min(options.threads, options.population);
    PopulationEvaluator<Genome, Agent> evaluator(options, population, scores, workers);
    long long bestEver = -1;
    bool baselineIncomplete = false;
    std::cout << "Local Space Invaders: " << options.population << " agents, " << workers
              << " workers, " << options.episodes << " seeds per evaluation\n" << std::flush;
    if (options.agent == AgentType::Cnn)
        std::cout << "CPU CNN: 32x30 pooled input -> 4x15x14 convolution -> 16 dense -> 6 actions; "
                  << CnnGenome::Parameters << " parameters, mutation rate=" << options.mutationRate
                  << " sigma=" << options.mutationSigma << '\n'
                  << "CNN inference: " << CnnBatch::backend() << "; up to " << cnnBatchSize(options)
                  << " games per worker; softmax sampling (temperature=1)\n" << std::flush;
    if (options.indefinite)
        std::cout << "Training indefinitely; press Ctrl+C to stop and keep the best checkpoint.\n" << std::flush;
    std::uint64_t generation = 0;
    for (;;)
    {
        if (interrupted(options)) break;
        const auto start = std::chrono::steady_clock::now();
        std::fill(scores.begin(), scores.end(), -1);
        // Simulation and interpretation are independent per worker. All mutation and
        // rand() calls stay on this thread, making results independent of scheduling.
        const long long totalFrames = evaluator.evaluate();
        // Keep checkpoint choice stable on ties, separately from parent selection.
        // In the first generation index zero is the explicitly/automatically loaded champion.
        const int champion = static_cast<int>(std::max_element(scores.begin(), scores.end()) - scores.begin());
        // Re-evaluate the loaded champion under the requested seeds/frame limit.
        // Keep its file intact on a tie; only an improvement should replace it.
        if (generation == 0 && continuingSavedBest && scores[0] >= 0) bestEver = scores[0];
        const long long best = scores[champion];
        // If a loaded champion has not completed its evaluation, no partial
        // generation can safely establish whether a candidate improves on it.
        baselineIncomplete = generation == 0 && !loadPath.empty() && scores[0] < 0;
        if (!baselineIncomplete && best >= 0 && best > bestEver)
        {
            saveTrainingBest(population[champion], options, best);
            bestEver = best;
        }
        const bool stopping = interrupted(options);
        const bool finalGeneration = !options.indefinite && generation + 1 == static_cast<std::uint64_t>(options.generations);
        if ((generation + 1) % 10 == 0 || finalGeneration || stopping)
        {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const int completed = static_cast<int>(std::count_if(scores.begin(), scores.end(), [](long long score) { return score >= 0; }));
            if (completed)
            {
                long long totalScore = 0;
                for (long long score : scores) if (score >= 0) totalScore += score;
                std::cout << "Generation " << generation + 1 << ": best=" << double(best) / options.episodes
                          << " mean=" << double(totalScore) / completed / options.episodes
                          << " frames/s=" << static_cast<long long>(totalFrames / seconds)
                          << " elapsed=" << seconds << 's';
                if (completed != options.population) std::cout << " completed=" << completed << '/' << options.population;
                std::cout << '\n' << std::flush;
            }
            else
                std::cout << "Generation " << generation + 1 << ": interrupted before any agent completed evaluation\n" << std::flush;
        }
        if (stopping || finalGeneration || interrupted(options)) break;
        rankInvadersPopulation(scores, order, tieRandom);
        std::vector<Genome> next;
        next.reserve(options.population);
        for (int i = 0; i < options.elite; ++i) next.push_back(population[order[i]]);
        for (int i = options.elite; i < options.population && !interrupted(options); ++i)
        {
            next.push_back(population[order[rand() % options.elite]]);
            mutate(next.back());
        }
        if (interrupted(options)) break;
        population.swap(next);
        ++generation;
    }
    if (interrupted(options)) std::cout << "Stopped after Ctrl+C.\n";
    if (bestEver >= 0)
        std::cout << "Best checkpoint: " << options.savePath << " (mean score "
                  << double(bestEver) / options.episodes << ")\n";
    else if (continuingSavedBest)
        std::cout << "Existing checkpoint retained: " << options.savePath << " (evaluation incomplete)\n";
    else if (baselineIncomplete)
        std::cout << "Loaded checkpoint evaluation was incomplete; no new checkpoint saved.\n";
    else
        std::cout << "No agent completed evaluation; no new checkpoint saved.\n";
}

static void trainCnnEs(const RunOptions& options)
{
    CnnEsOptimizer optimizer(options.population, options.es);
    bool continuingSavedBest = false;
    const std::string loadPath = trainingLoadPath(options, continuingSavedBest);
    continuingSavedBest |= !loadPath.empty() && loadPath == options.savePath;
    std::mt19937 random(options.seed);
    std::vector<CnnGenome> center(1), population(options.population);
    if (loadPath.empty())
    {
        center[0].randomize(random);
        std::cout << "Starting a fresh ES center\n";
    }
    else
    {
        center[0].load(loadPath);
        std::cout << "Starting from checkpoint: " << loadPath << "\nES Adam state starts fresh.\n";
    }
    std::vector<long long> scores(options.population, -1), centerScore(1, -1);
    const int workers = std::min(options.threads, options.population);
    PopulationEvaluator<CnnGenome, CnnAgent> evaluator(options, population, scores, workers);
    RunOptions centerOptions = options;
    centerOptions.population = centerOptions.threads = centerOptions.batchSize = 1;
    PopulationEvaluator<CnnGenome, CnnAgent> centerEvaluator(centerOptions, center, centerScore, 1);
    std::cout << "Local Space Invaders ES: " << options.population << " perturbations ("
              << options.population / 2 << " mirrored pairs), " << workers << " workers, "
              << options.episodes << " seeds per evaluation\n"
              << "CPU CNN: " << CnnGenome::Parameters << " parameters; centered ranks + Adam, sigma="
              << options.es.sigma << " learning-rate=" << options.es.learningRate
              << " weight-decay=" << options.es.weightDecay << '\n'
              << "CNN inference: " << CnnBatch::backend() << "; up to " << cnnBatchSize(options)
              << " games per worker; softmax sampling (temperature=1)\n"
              << "ES saves the best evaluated central policy; perturbations are never checkpointed.\n" << std::flush;
    if (options.indefinite)
        std::cout << "Training indefinitely; press Ctrl+C to stop and keep the best checkpoint.\n" << std::flush;

    long long bestEver = -1;
    const auto considerCenter = [&] {
        if (centerScore[0] >= 0 && centerScore[0] > bestEver)
        {
            saveTrainingBest(center[0], options, centerScore[0]);
            bestEver = centerScore[0];
        }
    };
    // Establish an evaluated baseline before allowing any optimizer update or save.
    centerEvaluator.evaluate();
    if (centerScore[0] >= 0)
    {
        if (continuingSavedBest) bestEver = centerScore[0];
        considerCenter();
        std::cout << "ES initial center: mean score " << double(centerScore[0]) / options.episodes << '\n' << std::flush;
    }
    for (std::uint64_t generation = 1; centerScore[0] >= 0 && !interrupted(options); ++generation)
    {
        const auto start = std::chrono::steady_clock::now();
        optimizer.perturb(center[0], random, population);
        if (interrupted(options)) break;
        if (generation == 1 || generation % 10 == 0)
            std::cout << "Evaluating ES generation " << generation << ": " << options.population
                      << " mirrored perturbations\n" << std::flush;
        std::fill(scores.begin(), scores.end(), -1);
        long long frames = evaluator.evaluate();
        const int completed = static_cast<int>(std::count_if(scores.begin(), scores.end(), [](long long s) { return s >= 0; }));
        if (completed != options.population || interrupted(options))
        {
            std::cout << "ES generation " << generation << ": update skipped; completed " << completed
                      << '/' << options.population << " perturbation evaluations\n" << std::flush;
            break;
        }
        const CnnEsStep step = optimizer.update(center[0], scores);
        // The updated center must actually play before it can become the best.
        centerScore[0] = -1;
        frames += centerEvaluator.evaluate();
        if (centerScore[0] < 0)
        {
            std::cout << "ES generation " << generation << ": center evaluation incomplete; checkpoint retained\n" << std::flush;
            break;
        }
        considerCenter();
        const bool last = !options.indefinite && generation == static_cast<std::uint64_t>(options.generations);
        if (generation == 1 || generation % 10 == 0 || last || interrupted(options))
        {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const long long total = std::accumulate(scores.begin(), scores.end(), 0LL);
            std::cout << "Generation " << generation << ": center=" << double(centerScore[0]) / options.episodes
                      << " best=" << double(bestEver) / options.episodes
                      << " mean=" << double(total) / options.population / options.episodes
                      << " perturbation-best=" << double(*std::max_element(scores.begin(), scores.end())) / options.episodes
                      << " gradient-norm=" << step.gradientNorm << " step-norm=" << step.stepNorm
                      << " frames/s=" << static_cast<long long>(frames / seconds)
                      << " elapsed=" << seconds << "s\n" << std::flush;
        }
        if (last) break;
    }
    if (interrupted(options)) std::cout << "Stopped after Ctrl+C.\n";
    if (bestEver >= 0)
        std::cout << "Best checkpoint: " << options.savePath << " (mean score " << double(bestEver) / options.episodes << ")\n";
    else if (continuingSavedBest)
        std::cout << "Existing checkpoint retained: " << options.savePath << " (evaluation incomplete)\n";
    else if (!loadPath.empty())
        std::cout << "Loaded checkpoint evaluation was incomplete; no new checkpoint saved.\n";
    else
        std::cout << "No agent completed evaluation; no new checkpoint saved.\n";
}

void trainInvaders(const RunOptions& options)
{
    InterruptHandler handler(options.indefinite);
    if (options.optimizer == TrainingOptimizer::Es)
    {
        if (options.agent != AgentType::Cnn) throw std::invalid_argument("ES requires a CNN agent");
        trainCnnEs(options);
    }
    else if (options.agent == AgentType::Cnn)
    {
        std::mt19937 random(options.seed);
        trainPopulation<CnnGenome, CnnAgent>(options,
            [&](CnnGenome& genome) { genome.randomize(random); },
            [&](CnnGenome& genome) { genome.mutate(random, options.mutationRate, options.mutationSigma); });
    }
    else
        trainPopulation<Genotype, InvadersAgent>(options, [](Genotype&) {},
                                                [](Genotype& genome) { genome.mutate(); });
}

template<class Action>
static void benchmarkPolicy(const RunOptions& options, Action action)
{
    SpaceInvaders game(options.seed, options.maxFrames);
    long long score = 0;
    int episodes = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int frame = 0; frame < options.benchmarkFrames; ++frame)
    {
        score += game.do_action(action(game, options.seed + static_cast<std::uint32_t>(episodes)));
        if (game.is_over()) game.reset(options.seed + static_cast<std::uint32_t>(++episodes));
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << options.benchmarkFrames << " frames in " << seconds << "s: "
              << static_cast<long long>(options.benchmarkFrames / seconds) << " frames/s; "
              << "episodes=" << episodes << " score=" << score << '\n';
}

void benchmarkInvaders(const RunOptions& options)
{
    if (options.agent == AgentType::Cnn)
    {
        CnnGenome genome;
        std::mt19937 random(options.seed);
        if (options.loadPath.empty()) genome.randomize(random);
        else genome.load(options.loadPath);
        CnnAgent agent(genome);
        std::cout << "CPU CNN (including pooled vision, inference, and softmax sampling): ";
        benchmarkPolicy(options, [&](SpaceInvaders& game, std::uint32_t seed) {
            if (game.world().frame == 0) agent.reset(seed);
            return agent.action(game);
        });
    }
    else if (!options.loadPath.empty())
    {
        Genotype genome;
        genome.load(options.loadPath);
        InvadersAgent agent(genome);
        benchmarkPolicy(options, [&](SpaceInvaders& game, std::uint32_t) { return agent.action(game); });
    }
    else
        benchmarkPolicy(options, [](SpaceInvaders& game, std::uint32_t) { game.getState(); return game.demoAction(); });
}
