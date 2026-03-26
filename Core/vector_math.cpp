#include "vector_math.h"
#include "advanced_math.h"

static inline int isNaN(double x) { return x != x; }
static inline int badSum(double s) { return (!(s > 0.0)) || isNaN(s) || (s > 1e308); }

namespace vector_math
{
    double vectorSum(const double *v, int n)
    {
        double sum = 0.0;
        for(int i = 0; i < n; i++) sum += v[i];
        return sum;
    }

    double vectorMax(const double *v, int n)
    {
        double m = v[0];
        for(int i = 1; i < n; i++)
            if(v[i] > m) m = v[i];
        return m;
    }

    double vectorDot(const double *a, const double *b, int n)
    {
        double sum = 0.0;
        for(int i = 0; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    void vectorScalar(double *v, int n, double scalar)
    {
        for(int i = 0; i < n; i++) v[i] *= scalar;
    }

    void vectorScalarDivide(double *v, int n, double scalar)
    {
        if(scalar == 0.0) return;
        for(int i = 0; i < n; i++) v[i] /= scalar;
    }

    void vectorAddition(const double *a, const double *b, double *result, int n)
    {
        for(int i = 0; i < n; i++) result[i] = a[i] + b[i];
    }

    void vectorSubtraction(const double *a, const double *b, double *result, int n)
    {
        for(int i = 0; i < n; i++) result[i] = a[i] - b[i];
    }

    void vectorCopy(const double *source, double *destination, int n)
    {
        for(int i = 0; i < n; i++) destination[i] = source[i];
    }

    void vectorFill(double *v, int n, double value)
    {
        for(int i = 0; i < n; i++) v[i] = value;
    }

    void softMax(const double *input, double *output, int n)
    {
        double maxValue = vectorMax(input, n);

        double sum = 0.0;
        for(int i = 0; i < n; i++)
        {
            double e = advanced_math::exponential(input[i] - maxValue);
            output[i] = e;
            sum += e;
        }

        // More robust check: handle underflow and overflow
        if(badSum(sum) || sum < 1e-300)
        {
            // All values went to zero (extreme underflow): use uniform distribution
            double u = 1.0 / (double)n;
            for(int i = 0; i < n; i++) output[i] = u;
            return;
        }

        double inv = 1.0 / sum;
        for(int i = 0; i < n; i++) output[i] *= inv;
    }

    void logSoftMax(const double *input, double *output, int n)
    {
        double maxValue = vectorMax(input, n);

        double sum = 0.0;
        for(int i = 0; i < n; i++)
            sum += advanced_math::exponential(input[i] - maxValue);

        if(badSum(sum))
        {
            double l = -advanced_math::logarithm((double)n);
            for(int i = 0; i < n; i++) output[i] = l;
            return;
        }

        double logSum = advanced_math::logarithm(sum);
        for(int i = 0; i < n; i++)
            output[i] = (input[i] - maxValue) - logSum;
    }
}
