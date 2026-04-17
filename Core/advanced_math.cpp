#include "advanced_math.h"

namespace advanced_math
{
double clamp(double value, double minimumValue, double maximumValue)
{
    if (minimumValue > maximumValue) {
        double temporary = minimumValue;
        minimumValue = maximumValue;
        maximumValue = temporary;
    }

    if (value < minimumValue) {
        return minimumValue;
    }
    if (value > maximumValue) {
        return maximumValue;
    }
    return value;
}

double exponential(double value)
{
    value = clamp(value, -40.0, 40.0);
    double sum = 1.0;
    double term = 1.0;

    // Use adaptive convergence: stop when term becomes negligible (< 1e-15)
    // or after max 50 iterations to be safe
    for (int n = 1; n <= 50; n++) {
        term = term * value / n;
        sum += term;
        if (term < 1e-15 && term > -1e-15) {
            break;
        }
    }
    return sum;
}

double sigmoid(double value)
{
    value = clamp(value, -20.0, 20.0);
    if (value >= 0.0) {
        double x = exponential(-value);
        return 1.0 / (1.0 + x);
    }

    {
        double x = exponential(value);
        return x / (1.0 + x);
    }
}

double sigmoidDeriv(double value)
{
    double y = sigmoid(value);
    return y * (1 - y);
}

double tanh(double value)
{
    value = clamp(value, -20.0, 20.0);
    double posExpo = exponential(value);
    double negExpo = exponential(-value);
    return (posExpo - negExpo) / (posExpo + negExpo);
}

double tanhDeriv(double value)
{
    double y = tanh(value);
    return 1 - (y * y);
}

double reLu(double value)
{
    if (value > 0.0) {
        return value;
    }
    return 0.0;
}

double reLuDeriv(double value)
{
    if (value > 0.0) {
        return 1.0;
    }
    return 0.0;
}

double logarithm(double value)
{
    if (value <= 0.0) {
        return -1e9;
    }

    double e = exponential(1.0);
    double sqrtE = exponential(0.5);
    int k = 0;

    // Reduce argument to [1/sqrt(e), sqrt(e)] for fast Taylor convergence
    while (value > sqrtE) {
        value /= e;
        k++;
    }
    while (value < 1.0 / sqrtE) {
        value *= e;
        k--;
    }

    // ln(1+n), n in [-0.39, 0.65]: converges well with adaptive checking.
    double n = value - 1.0;
    double sum = 0.0;
    double term = n;

    for (int i = 1; i <= 50; i++) {
        sum += term / (double)i;
        term *= -n;
        if (term < 1e-15 && term > -1e-15) {
            break;
        }
    }

    return sum + k;
}

double squareRoot(double value)
{
    if (value < 0.0) {
        return -1e9;
    }
    if (value == 0.0) {
        return 0.0;
    }

    // Smart initial guess for all ranges
    double x;
    if (value >= 1.0) {
        x = value;
    } else if (value >= 1e-4) {
        x = 1.0;
    } else {
        x = 1.0 / value;
    }

    for (int i = 0; i < 30; i++) {
        double xNew = 0.5 * (x + (value / x));
        if (xNew == x) {
            break;
        }
        x = xNew;
    }
    return x;
}
}