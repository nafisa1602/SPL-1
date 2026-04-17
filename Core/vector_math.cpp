#include "vector_math.h"
#include "advanced_math.h"

static inline int isNaN(double x) { return x != x; }
static inline int badSum(double s) { return (!(s > 0.0)) || isNaN(s) || (s > 1e308); }

namespace vector_math
{
double vectorSum(const double* values, int size)
{
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += values[i];
    }
    return sum;
}

double vectorMax(const double* values, int size)
{
    double maxValue = values[0];
    for (int i = 1; i < size; i++) {
        if (values[i] > maxValue) {
            maxValue = values[i];
        }
    }
    return maxValue;
}

double vectorDot(const double* lhs, const double* rhs, int size)
{
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

void vectorScalar(double* values, int size, double scalar)
{
    for (int i = 0; i < size; i++) {
        values[i] *= scalar;
    }
}

void vectorScalarDivide(double* values, int size, double scalar)
{
    if (scalar == 0.0) {
        return;
    }

    for (int i = 0; i < size; i++) {
        values[i] /= scalar;
    }
}

void vectorAddition(const double* lhs, const double* rhs, double* result, int size)
{
    for (int i = 0; i < size; i++) {
        result[i] = lhs[i] + rhs[i];
    }
}

void vectorSubtraction(const double* lhs, const double* rhs, double* result, int size)
{
    for (int i = 0; i < size; i++) {
        result[i] = lhs[i] - rhs[i];
    }
}

void vectorCopy(const double* source, double* destination, int size)
{
    for (int i = 0; i < size; i++) {
        destination[i] = source[i];
    }
}

void vectorFill(double* values, int size, double value)
{
    for (int i = 0; i < size; i++) {
        values[i] = value;
    }
}

void softMax(const double* input, double* output, int size)
{
    double maxValue = vectorMax(input, size);
    double sum = 0.0;

    for (int i = 0; i < size; i++) {
        double e = advanced_math::exponential(input[i] - maxValue);
        output[i] = e;
        sum += e;
    }

    if (badSum(sum) || sum < 1e-300) {
        // Extreme underflow fallback: preserve a valid probability distribution.
        double uniform = 1.0 / (double)size;
        for (int i = 0; i < size; i++) {
            output[i] = uniform;
        }
        return;
    }

    double invSum = 1.0 / sum;
    for (int i = 0; i < size; i++) {
        output[i] *= invSum;
    }
}

void logSoftMax(const double* input, double* output, int size)
{
    double maxValue = vectorMax(input, size);
    double sum = 0.0;

    for (int i = 0; i < size; i++) {
        sum += advanced_math::exponential(input[i] - maxValue);
    }

    if (badSum(sum)) {
        double logUniform = -advanced_math::logarithm((double)size);
        for (int i = 0; i < size; i++) {
            output[i] = logUniform;
        }
        return;
    }

    double logSum = advanced_math::logarithm(sum);
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] - maxValue) - logSum;
    }
}
}
