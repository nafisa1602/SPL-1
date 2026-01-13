#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H
#include "vector_math.h"
namespace matrix_math
{
void matrixZero(double *a, int row, int column);
void matrixCopy(const double *source, double *dest, int row, int column);
void matrixPrint(const double *a, int row, int column);
void matrixAdd(const double *a, const double *b, double *c, int row, int column);
void matrixSubtract(const double *a, const double *b, double *c, int row, int column);
void matrixScalarMultiply(const double *a, int row, int column, double scalar);
bool matrixMultiply(const double *a, const double *b, int rowA, int colA, int rowB, int colB, double *c);
void matrixTranspose(const double *a, int row, int column, double *at);
void matrixRowSum(const double *a, double *result, int row, int column);
void matrixRowMax(const double *a, double *result, int row, int column);
void matrixExpo(double *a, int row, int column);
void matrixLog(double *a, int row, int column);
void matrixAddRowVec(double *a, const double *v, int row, int column);
void matrixDivRowVec(double *a, const double *v, int row, int column);
}
#endif 