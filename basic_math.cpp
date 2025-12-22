#include "basic_math.h"
namespace basic_math
{
double absolute(double number)
{
    if(number < 0.0) return -number;
    return number;
}
double  minimum(double a, double b)
{
    if(a < b) return a;
    return b;
}
double  maximum(double a, double b)
{
    if(a > b) return a;
    return b;
}
long double factorial(double number)
{
    if(number < 0.0) return 0.0L;
    long double result = 1.0L;
    for(int i = 2; i <= (int)number; i++)
    {
        result *= i;
    }
    return result;
}
double power(double base, int exponent)
{
    if(exponent == 0) return 1.0;
    else if(exponent < 0) return 1.0 / power(base, -exponent);
    double result = 1.0;
    while(exponent > 0)
    {
        if(exponent % 2 == 1) result *= base;
        base *= base;
        exponent /= 2;
    }
    return result;
}
}