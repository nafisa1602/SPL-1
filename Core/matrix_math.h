#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#include "vector_math.h"

namespace matrix_math
{
void matrixZero(double* matrix, int rows, int cols);
void matrixCopy(const double* source, double* destination, int rows, int cols);
void matrixPrint(const double* matrix, int rows, int cols);
void matrixAdd(const double* lhs, const double* rhs, double* result, int rows, int cols);
void matrixSubtract(const double* lhs, const double* rhs, double* result, int rows, int cols);
void matrixScalarMultiply(double* matrix, int rows, int cols, double scalar);
bool matrixMultiply(const double* lhs, const double* rhs, int lhsRows, int lhsCols, int rhsRows, int rhsCols, double* result);
void matrixTranspose(const double* matrix, int rows, int cols, double* transpose);
void matrixRowSum(const double* matrix, double* result, int rows, int cols);
void matrixRowMax(const double* matrix, double* result, int rows, int cols);
void matrixExpo(double* matrix, int rows, int cols);
void matrixLog(double* matrix, int rows, int cols);
void matrixAddRowVec(double* matrix, const double* rowVector, int rows, int cols);
void matrixDivRowVec(double* matrix, const double* rowVector, int rows, int cols);
}

#endif 