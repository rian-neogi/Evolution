#include "Genotype.h"
#include <fstream>
#include <stdexcept>
#include <limits>

const int CLONE_CHANCE = 5;

Genotype::Genotype()
{
    mGenes.push_back(Gene());
}

Genotype::~Genotype()
{
    
}

void Genotype::mutate()
{
    if(rand()%100 < CLONE_CHANCE)
    {
        mGenes.push_back(Gene(mGenes[rand()%mGenes.size()]));
    }
    for(int i = 0;i<mGenes.size();i++)
    {
        int x = rand()%MUT_NUM;
        mGenes[i].mutate(x);    
    }
}

void Genotype::save(const std::string& path) const
{
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Cannot open checkpoint for writing: " + path);
    file << mGenes.size() << '\n';
    for (const auto& gene : mGenes)
    {
        file << gene.mCode.size() << '\n';
        for (const auto& instruction : gene.mCode)
            file << instruction.type << '\n' << instruction.arg1 << '\n' << instruction.arg2 << '\n';
    }
    file.close();
    if (!file) throw std::runtime_error("Failed to write checkpoint: " + path);
}

void Genotype::load(const std::string& path)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open checkpoint: " + path);
    const auto invalid = [&]() { return std::runtime_error("Invalid checkpoint: " + path); };
    long long count;
    if (!(file >> count) || count < 1 || count > 10000) throw invalid();
    std::vector<Gene> genes;
    genes.reserve(static_cast<std::size_t>(count));
    long long total = 0;
    for (long long i = 0; i < count; ++i)
    {
        long long instructions;
        if (!(file >> instructions) || instructions < 1 || instructions > 100000 ||
            (total += instructions) > 1000000) throw invalid();
        genes.emplace_back();
        auto& code = genes.back().mCode;
        code.clear();
        code.reserve(static_cast<std::size_t>(instructions));
        for (long long j = 0; j < instructions; ++j)
        {
            long long type, arg1, arg2;
            if (!(file >> type >> arg1 >> arg2) || type < 0 || type >= INST_NUM ||
                arg1 < 0 || arg1 > std::numeric_limits<int>::max() ||
                arg2 < 0 || arg2 > std::numeric_limits<int>::max()) throw invalid();
            code.emplace_back(static_cast<int>(type), static_cast<int>(arg1), static_cast<int>(arg2));
        }
    }
    file >> std::ws;
    if (!file.eof()) throw invalid();
    mGenes.swap(genes);
}
