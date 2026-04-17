#include "basic_math.h"

namespace basic_math
{
double absolute(double value)
{
    if (value < 0.0) {
        return -value;
    }
    return value;
}

double minimum(double lhs, double rhs)
{
    if (lhs < rhs) {
        return lhs;
    }
    return rhs;
}

double maximum(double lhs, double rhs)
{
    if (lhs > rhs) {
        return lhs;
    }
    return rhs;
}

long double factorial(int value)
{
    if (value < 0) {
        return 0.0L;
    }

    long double result = 1.0L;
    for (int i = 2; i <= value; i++) {
        result *= i;
    }
    return result;
}

double power(double base, int exponent)
{
    if (exponent == 0) {
        return 1.0;
    }
    if (exponent < 0) {
        return 1.0 / power(base, -exponent);
    }

    double result = 1.0;
    while (exponent > 0) {
        if (exponent % 2 == 1) {
            result *= base;
        }
        base *= base;
        exponent /= 2;
    }
    return result;
}
}