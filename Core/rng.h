#ifndef RNG_H
#define RNG_H
namespace rng
{
    void seed(unsigned int s);
    unsigned int shift();
    double uniform01();
    double uniform(double a, double b);
}
#endif