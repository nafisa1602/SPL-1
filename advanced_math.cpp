#include "advanced_math.h"
namespace advanced_math
{
double clamp(double number, double lowest, double highest)
{
    if(lowest > highest)
    {
        double temporary = lowest;
        lowest = highest;
        highest = temporary;
    }
    if(number < lowest) return lowest;
    else if(number > highest) return highest;
    else return number;
}
double exponential(double number)
{
    number = clamp(number, -40.0, 40.0);
    double sum = 1.0;
    double term = 1.0;
    for(int n = 1; n <= 20; n++)
    {
        term = term*number/n;
        sum += term;
    }
    return sum;
}
double sigmoid(double number)
{
    number = clamp(number, -20.0, 20.0);
    if(number >= 0.0)
    {
        double x = exponential(-number);
        return 1.0 / (1.0 + x);
    }
    else
    {
        double x = exponential(number);
        return 1.0 / (1.0 + x);
    }
}
double tanh(double number)
{
   number = clamp(number, -20.0, 20.0);
   double posExpo = exponential(number);
   double negExpo = exponential(-number);
   return (posExpo - negExpo) / (posExpo + negExpo);
}
double logarithm(double number)
{
    if(number <= 0.0) return -1e9;
    double e = exponential(1.0);
}
}