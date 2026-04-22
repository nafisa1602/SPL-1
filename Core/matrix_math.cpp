#include <iostream>
#include "advanced_math.h"
#include "matrix_math.h"

namespace matrix_math
{

void matrixZero(double* matrix, int rows, int cols)
{
    for (int r = 0; r < rows; r++) 
    {
        for (int c = 0; c < cols; c++) 
        {
            matrix[r * cols + c] = 0.0;
        }
    }
}

void matrixCopy(const double* source, double* destination, int rows, int cols)
{
    for (int r = 0; r < rows; r++) 
    {
        for (int c = 0; c < cols; c++) 
        {
            destination[r * cols + c] = source[r * cols + c];
        }
    }
}

void matrixPrint(const double* matrix, int rows, int cols)
{
    for (int r = 0; r < rows; r++) 
    {
        for (int c = 0; c < cols; c++) 
        {
            std::cout << matrix[r * cols + c] << " ";
        }
        std::cout << std::endl;
    }
}

void matrixAdd(const double* lhs, const double* rhs, double* result, int rows, int cols)
{
    for (int r = 0; r < rows; r++) 
    {
        for (int c = 0; c < cols; c++) 
        {
            result[r * cols + c] = lhs[r * cols + c] + rhs[r * cols + c];
        }
    }
}

void matrixSubtract(const double* lhs, const double* rhs, double* result, int rows, int cols)
{
    for (int r = 0; r < rows; r++) 
    {
        for (int c = 0; c < cols; c++) 
        {
            result[r * cols + c] = lhs[r * cols + c] - rhs[r * cols + c];
        }
    }
}

void matrixScalarMultiply(double* matrix, int rows, int cols, double scalar)
{
    for (int r = 0; r < rows; r++) 
    {
        for (int c = 0; c < cols; c++) 
        {
            matrix[r * cols + c] = matrix[r * cols + c] * scalar;
        }
    }
}

bool matrixMultiply(const double* lhs, const double* rhs, int lhsRows, int lhsCols, int rhsRows, int rhsCols, double* result)
{
    if (lhsCols != rhsRows) 
    {
        return false;
    }
    if (lhs == nullptr || rhs == nullptr || result == nullptr) 
    {
        return false;
    }
    for (int r = 0; r < lhsRows; r++) 
    {
        for (int c = 0; c < rhsCols; c++) 
        {
            result[r * rhsCols + c] = 0.0;
            for (int k = 0; k < lhsCols; k++) 
            {
                result[r * rhsCols + c] += lhs[r * lhsCols + k] * rhs[k * rhsCols + c];
            }
        }
    }
    return true;
}

void matrixTranspose(const double* matrix, int rows, int cols, double* transpose)
{
    for (int r = 0; r < rows; r++) 
    {
        for (int c = 0; c < cols; c++) 
        {
            transpose[c * rows + r] = matrix[r * cols + c];
        }
    }
}

void matrixRowSum(const double* matrix, double* result, int rows, int cols)
{
    for (int r = 0; r < rows; r++) 
    {
        double rowSum = 0.0;
        for (int c = 0; c < cols; c++) 
        {
            rowSum += matrix[r * cols + c];
        }
        result[r] = rowSum;
    }
}

void matrixRowMax(const double* matrix, double* result, int rows, int cols)
{
    for (int r = 0; r < rows; r++) 
    {
        double maxValue = matrix[r * cols];
        for (int c = 0; c < cols; c++) 
        {
            double value = matrix[r * cols + c];
            if (maxValue < value) 
            {
                maxValue = value;
            }
        }
        result[r] = maxValue;
    }
}

void matrixExpo(double* matrix, int rows, int cols)
{
    for (int r = 0; r < rows; r++) 
    {
        for (int c = 0; c < cols; c++) 
        {
            matrix[r * cols + c] = advanced_math::exponential(matrix[r * cols + c]);
        }
    }
}

void matrixLog(double* matrix, int rows, int cols)
{
    for (int r = 0; r < rows; r++) 
    {
        for (int c = 0; c < cols; c++) 
        {
            matrix[r * cols + c] = advanced_math::logarithm(matrix[r * cols + c]);
        }
    }
}

void matrixAddRowVec(double* matrix, const double* rowVector, int rows, int cols)
{
    for (int r = 0; r < rows; r++) 
    {
        for (int c = 0; c < cols; c++) 
        {
            matrix[r * cols + c] += rowVector[c];
        }
    }
}

void matrixDivRowVec(double* matrix, const double* rowVector, int rows, int cols)
{
    for (int r = 0; r < rows; r++) 
    {
        for (int c = 0; c < cols; c++) 
        {
            if (rowVector[c] != 0.0) 
            {
                matrix[r * cols + c] /= rowVector[c];
            }
        }
    }
}
}