#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H

namespace vector_math
{
double vectorSum(const double* values, int size);
double vectorMax(const double* values, int size);
double vectorDot(const double* lhs, const double* rhs, int size);

void vectorScalar(double* values, int size, double scalar);
void vectorScalarDivide(double* values, int size, double scalar);

void vectorAddition(const double* lhs, const double* rhs, double* result, int size);
void vectorSubtraction(const double* lhs, const double* rhs, double* result, int size);

void vectorCopy(const double* source, double* destination, int size);
void vectorFill(double* values, int size, double value);

void softMax(const double* input, double* output, int size);
void logSoftMax(const double* input, double* output, int size);
}

#endif
