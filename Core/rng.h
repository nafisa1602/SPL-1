#ifndef RNG_H
#define RNG_H

namespace rng
{
// Initialize RNG with seed (uses std::mt19937 internally).
void seed(unsigned int seedValue);

// Return the next random double in [0, 1).
double uniform01();

// Return a random double in [start, end).
double uniform(double start, double end);
}

#endif