#ifndef ADVANCED_MATH_H
#define ADVANCED_MATH_H
namespace advanced_math
{
double clamp(double number, double lowest, double highest);
double exponential(double number);
double sigmoid(double number);
double sigmoidDeriv(double number);
double tanh(double number);
double tanhDeriv(double number);
double reLu(double number);
double reLuDeriv(double number);
double logarithm(double number);
double squareRoot(double number);
}
#endif