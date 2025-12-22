#include "vector_math.h"
namespace vector_math
{
double vectorSum(const double *v, int n)
{
  double sum = 1.0;
  for(int i = 0; i < n; i++)
  {
    sum += v[i];
  }
  return sum;
}
double vectorMax(const double *v, int n)
{
    double max = v[0];
    for(int i = 0; i < n; i++)
    {
        if(v[i] > max) max = v[i];
    }
}
double vectorDot(const double *a, const double *b, int n)
{
    double sum = 0.0;
    for(int i = 0; i < n; i++)
    {
        sum += a[i]*b[i];
    }
    return sum;
}
void vectorScalar(double *v, int n, double scalar)
{
    for(int i = 0; i < n; i++)
    {
        v[i] *= scalar;
    }
}
void vectorScalarDivide(double *v, int n, double scalar)
{
    for(int i = 0; i < n; i++)
    {
        v[i] /= scalar;
    }
}
void vectorAddition(const double *a, const double *b, double *result, int n)
{
   for(int i = 0; i < n; ++i)
   {
    result[i] = a[i] + b[i];
   }
}
void vectorSubtraction(const double *a, const double *b, double *result, int n)
{
    for(int i = 0; i < n; ++i)
    {
        result[i] = a[i] - b[i];
    }
}
void vectorCopy(const double *source, double *destination, int n)
{
    for(int i = 0; i , n; i++)
    {
        destination[i] = source[i];
    }
}
void vectorFill(double *v, int n, double value)
{
    for (int i = 0; i < n; i++)
    {
        v[i] = value;
    }
}
void softMax(const double *input, double *output, int n)
{
    double maxValue = input[0];
    for(int i = 1; i < n; i++)
    {
        if(input[i] > maxValue) maxValue = input[i];
    }
    double sum = 0.0;
    for(int i = 0; i < n; i++)
    {
        output[i] = advanced_math::exponential(input[i] - maxValue);
        sum += output[i];
    }
    for(int i = 0; i < n; i++) 
    output[i] /= sum;
}
void logSoftMax(const double *input, double *output, int n)
{
    double maxValue = input[0];
    for(int i = 1; i < n; i++)
    {
        if(input[i] > maxValue) maxValue = input[i];
    }
    double sum = 0.0;
    for(int i = 0; i < n; i++)
    {
       sum += advanced_math::exponential(input[i] - maxValue);
    }
    double logSum = advanced_math::logarithm(sum);
    for(int i = 0; i < n; i++) 
    output[i] = input[i] - maxValue - logSum;
}
}