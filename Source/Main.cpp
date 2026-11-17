#include "LocalTraining.h"
#include "CnnAgent.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <thread>

namespace {
void usage()
{
    std::cout << "Local Space Invaders\n"
              << "  Evolution train [--generations 100 | --indefinite] [--population 128] [--elite 8]\n"
              << "                  [--threads N] [--episodes 3] [--save invaders_best] [--load FILE | --fresh]\n"
              << "                  Loads the save checkpoint automatically when it exists; --fresh starts over.\n"
              << "                  --threads defaults to available logical CPUs; use 1 for serial training.\n"
              << "                  --indefinite runs until Ctrl+C, retaining the best completed evaluation.\n"
              << "                  [--agent program|cnn] [--mutation-rate 0.1] [--mutation-sigma 0.05]\n"
              << "                  CNN uses CPU inference and evolutionary weights; default save: invaders_cnn_best.\n"
              << "                  [--batch-size 4] CNN games per worker (maximum); 1 disables game batching.\n"
              << "                  [--optimizer ga|es] defaults to ga; es is CNN-only, with an even population.\n"
              << "                  ES: [--es-sigma 0.02] [--learning-rate 0.01] [--weight-decay 0.005]\n"
              << "                  ES uses mirrored noise, centered ranks, and Adam; saves the best central policy.\n"
              << "                  --elite and --mutation-* apply only to ga. Checkpoints contain weights, not Adam state.\n"
              << "  Evolution watch [--load FILE] [--fps 60] [--episodes 1]\n"
              << "                  Detects agent type from --load; --agent cnn alone watches invaders_cnn_best.\n"
              << "                  Without either option, shows the built-in demonstration pilot.\n"
              << "  Evolution play  [--fps 60] [--episodes 1]\n"
              << "  Evolution benchmark [--frames 1000000]\n"
              << "                      [--agent cnn] [--load FILE] includes the selected agent's inference.\n"
              << "  Common: [--seed 123] [--max-frames 3600]\n"
              << "  Viewer: arrows/A/D move, Space fires; P pauses, N steps, R restarts, Esc exits.\n"
              << "  Training is headless and uses no ROM or ALE. Program agents read RAM; CNN agents read pooled pixels.\n";
}

int number(const std::string& value, int minimum, int maximum)
{
    std::size_t end = 0;
    const long long n = std::stoll(value, &end);
    if (end != value.size() || n < minimum || n > maximum)
        throw std::invalid_argument("Number outside supported range: " + value);
    return static_cast<int>(n);
}

float boundedFloat(const std::string& value, float maximum, bool allowZero = false)
{
    std::size_t end = 0;
    const float n = std::stof(value, &end);
    if (end != value.size() || !std::isfinite(n) || (allowZero ? n < 0 : n <= 0) || n > maximum)
        throw std::invalid_argument("Number outside supported range: " + value);
    return n;
}
}

int main(int argc, char** argv)
{
    try
    {
        if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")
        {
            usage();
            return 0;
        }
        const std::string mode = argv[1];
        if (mode != "train" && mode != "watch" && mode != "play" && mode != "benchmark")
            throw std::invalid_argument("Unknown command: " + mode);
        RunOptions options;
        options.threads = std::max(1u, std::thread::hardware_concurrency());
        if (mode != "train") options.episodes = 1;
        bool explicitElite = false, explicitAgent = false, explicitSave = false, cnnMutation = false;
        bool explicitGenerations = false, explicitBatch = false;
        bool explicitOptimizer = false, esSettings = false;
        for (int i = 2; i < argc; ++i)
        {
            const std::string flag = argv[i];
            if (flag == "--help") { usage(); return 0; }
            if (flag == "--fresh")
            {
                if (mode != "train") throw std::invalid_argument("--fresh is only available for training");
                options.fresh = true;
                continue;
            }
            if (flag == "--indefinite")
            {
                if (mode != "train") throw std::invalid_argument("--indefinite is only available for training");
                options.indefinite = true;
                continue;
            }
            if (++i >= argc) throw std::invalid_argument("Missing value for " + flag);
            const std::string value = argv[i];
            if (flag == "--seed") options.seed = number(value, 0, 2147483647);
            else if (flag == "--max-frames") options.maxFrames = number(value, 1, 10000000);
            else if (flag == "--episodes") options.episodes = number(value, 1, 10000);
            else if (flag == "--population") options.population = number(value, 2, 100000);
            else if (flag == "--elite") { options.elite = number(value, 1, 100000); explicitElite = true; }
            else if (flag == "--generations") { options.generations = number(value, 1, 10000000); explicitGenerations = true; }
            else if (flag == "--threads") options.threads = number(value, 1, 256);
            else if (flag == "--batch-size") { options.batchSize = number(value, 1, 256); explicitBatch = true; }
            else if (flag == "--fps") options.fps = number(value, 1, 1000);
            else if (flag == "--frames") options.benchmarkFrames = number(value, 1, 1000000000);
            else if (flag == "--load") options.loadPath = value;
            else if (flag == "--save") { options.savePath = value; explicitSave = true; }
            else if (flag == "--agent")
            {
                if (value != "program" && value != "cnn")
                    throw std::invalid_argument("--agent must be program or cnn");
                options.agent = value == "cnn" ? AgentType::Cnn : AgentType::Program;
                explicitAgent = true;
            }
            else if (flag == "--mutation-rate")
            {
                options.mutationRate = boundedFloat(value, 1);
                cnnMutation = true;
            }
            else if (flag == "--mutation-sigma")
            {
                options.mutationSigma = boundedFloat(value, 8);
                cnnMutation = true;
            }
            else if (flag == "--optimizer")
            {
                if (value != "ga" && value != "es")
                    throw std::invalid_argument("--optimizer must be ga or es");
                options.optimizer = value == "es" ? TrainingOptimizer::Es : TrainingOptimizer::Genetic;
                explicitOptimizer = true;
            }
            else if (flag == "--es-sigma")
            {
                options.es.sigma = boundedFloat(value, 8);
                esSettings = true;
            }
            else if (flag == "--learning-rate")
            {
                options.es.learningRate = boundedFloat(value, 1);
                esSettings = true;
            }
            else if (flag == "--weight-decay")
            {
                options.es.weightDecay = boundedFloat(value, 1, true);
                esSettings = true;
            }
            else throw std::invalid_argument("Unknown option: " + flag);
        }
        if (!explicitElite) options.elite = std::max(1, options.population / 16);
        if (options.fresh && !options.loadPath.empty())
            throw std::invalid_argument("Use either --fresh or --load, not both");
        if (options.indefinite && explicitGenerations)
            throw std::invalid_argument("Use either --indefinite or --generations, not both");
        const auto detectAgent = [&](const std::string& path) {
            const auto detected = isCnnCheckpoint(path) ? AgentType::Cnn : AgentType::Program;
            if (explicitAgent && detected != options.agent)
                throw std::invalid_argument("Checkpoint does not match --agent: " + path);
            options.agent = detected;
        };
        if (!options.loadPath.empty()) detectAgent(options.loadPath);
        if (!explicitSave && options.agent == AgentType::Cnn) options.savePath = "invaders_cnn_best";
        if (mode == "train" && options.loadPath.empty() && !options.fresh)
        {
            struct stat info;
            if (stat(options.savePath.c_str(), &info) == 0) detectAgent(options.savePath);
        }
        if (mode == "watch" && options.agent == AgentType::Cnn && options.loadPath.empty())
        {
            options.loadPath = options.savePath;
            detectAgent(options.loadPath);
        }
        if (cnnMutation && (mode != "train" || options.agent != AgentType::Cnn))
            throw std::invalid_argument("CNN mutation options are only available for CNN training");
        if (explicitBatch && (mode != "train" || options.agent != AgentType::Cnn))
            throw std::invalid_argument("--batch-size is only available for CNN training");
        if (explicitOptimizer && mode != "train")
            throw std::invalid_argument("--optimizer is only available for training");
        if (esSettings && (mode != "train" || options.optimizer != TrainingOptimizer::Es))
            throw std::invalid_argument("ES settings require train --optimizer es");
        if (options.optimizer == TrainingOptimizer::Es)
        {
            if (options.agent != AgentType::Cnn)
                throw std::invalid_argument("--optimizer es requires a CNN agent (--agent cnn or a CNN checkpoint)");
            if (options.population % 2)
                throw std::invalid_argument("ES requires an even --population (mirrored perturbation pairs)");
            if (explicitElite || cnnMutation)
                throw std::invalid_argument("--elite and --mutation-* are only available with --optimizer ga");
        }
        else if (options.elite >= options.population)
            throw std::invalid_argument("--elite must be smaller than --population");
        if (mode == "train") trainInvaders(options);
        else if (mode == "benchmark") benchmarkInvaders(options);
        else return viewInvaders(options, mode == "play");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
