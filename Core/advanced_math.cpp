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
    // Use adaptive convergence: stop when term becomes negligible (< 1e-15)
    // or after max 50 iterations to be safe
    for(int n = 1; n <= 50; n++)
    {
        term = term * number / n;
        sum += term;
        // Stop when term is negligible
        if(term < 1e-15 && term > -1e-15)
            break;
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
        return x / (1.0 + x);
    }
}
double sigmoidDeriv(double number)
{
    double value = sigmoid(number);
    return value * (1 - value);
}
double tanh(double number)
{
   number = clamp(number, -20.0, 20.0);
   double posExpo = exponential(number);
   double negExpo = exponential(-number);
   return (posExpo - negExpo) / (posExpo + negExpo);
}
double tanhDeriv(double number)
{
    double value = tanh(number);
    return 1 - (value * value);
}
double reLu(double number)
{
    if(number > 0.0) return number;
    return 0.0;
}
double reLuDeriv(double number)
{
   if(number > 0.0) return 1.0;
   return 0.0;
}
double logarithm(double number)
{
    if(number <= 0.0) return -1e9;
    double e = exponential(1.0);
    double sqrtE = exponential(0.5);
    int k = 0;
    // Reduce argument to [1/sqrt(e), sqrt(e)] for fast Taylor convergence
    while(number > sqrtE)  { number /= e; k++; }
    while(number < 1.0 / sqrtE) { number *= e; k--; }
    // ln(1+n), n in [-0.39, 0.65]: converges well with adaptive checking
    double n = number - 1.0;
    double sum = 0.0;
    double term = n;
    for(int i = 1; i <= 50; i++)
    {
        sum += term / (double)i;
        term *= -n;
        // Stop when term becomes negligible
        if(term < 1e-15 && term > -1e-15)
            break;
    }
    return sum + k;
}
double squareRoot(double number)
{
    if(number < 0.0) return -1e9;
    else if(number == 0.0) return 0.0;
    // Smart initial guess for all ranges
    double x;
    if(number >= 1.0)       x = number;
    else if(number >= 1e-4) x = 1.0;
    else                    x = 1.0 / number;
    for(int i = 0; i < 30; i++)
    {
        double xNew = 0.5 * (x + (number / x));
        if(xNew == x) break;
        x = xNew;
    }
    return x;
}
}