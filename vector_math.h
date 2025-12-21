#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H
namespace vector_math
{
double vectorSum(const double *v, int n);
double vectorMax(const double *v, int n);
double vectorDot(const double *a, const double *b, int n);
void vectorScalar(double *v, int n, double scalar);
void vectorAddition(const double *a, const double *b, double *result, int n);
void vectorSubtraction(const double *a, const double *b, double *result, int n);
void vectorCopy(const double *source, const double *destination, int n);
}
#endif