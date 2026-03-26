#include "rng.h"
#include <random>

namespace rng
{
    // Use MT19937 for better quality RNG with period 2^19937-1 (vs XorShift32's 2^32)
    static std::mt19937 generator(2463534242u);
    
    void seed(unsigned int s)
    {
        if(s == 0u) 
            generator.seed(2463534242u);
        else 
            generator.seed(s);
    }
    
    double uniform01()
    {
        // Generate uniform random in [0.0, 1.0)
        static std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(generator);
    }
    
    double uniform(double a, double b)
    {
        return a + (b - a) * uniform01();
    }
}