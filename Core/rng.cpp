#include"rng.h"
namespace rng
{
    static unsigned int state = 2463534242u;
    void seed(unsigned int s)
    {
        if(s == 0u) state = 2463534242u;
        else state = s;
    }
    unsigned int shift()
    {
        unsigned int x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }
    double uniform01()
    {
        unsigned int x = shift();
        return (x >> 8) * (1.0 / 16777216.0);
    }
    double uniform(double a, double b)
    {
        return a + (b - a) * uniform01();
    }
}