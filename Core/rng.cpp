#include "rng.h"
#include <random>

namespace rng
{
// Use MT19937 for stable, reproducible pseudo-random generation.
static std::mt19937 generator(2463534242u);

void seed(unsigned int seedValue)
{
    if (seedValue == 0u) {
        generator.seed(2463534242u);
        return;
    }

    generator.seed(seedValue);
}

double uniform01()
{
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(generator);
}

double uniform(double start, double end)
{
    return start + (end - start) * uniform01();
}
}