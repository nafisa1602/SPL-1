#ifndef ADVANCED_MATH_H
#define ADVANCED_MATH_H

namespace advanced_math
{
double clamp(double value, double minimumValue, double maximumValue);
double exponential(double value);
double sigmoid(double value);
double sigmoidDeriv(double value);
double tanh(double value);
double tanhDeriv(double value);
double reLu(double value);
double reLuDeriv(double value);
double logarithm(double value);
double squareRoot(double value);
}

#endif