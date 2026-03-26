#ifndef RNG_H
#define RNG_H
namespace rng
{
    // Initialize RNG with seed (use std::mt19937 internally for better quality)
    void seed(unsigned int s);
    // Get next random double in [0, 1)
    double uniform01();
    // Get random double in [a, b)
    double uniform(double a, double b);
}
#endif