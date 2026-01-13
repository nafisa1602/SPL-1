#include<iostream>
#include "matrix_math.h"
namespace matrix_math
{
void matrixZero(double *a, int row, int column)
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < column; j++)
        {
            a[ i*column + j ] = 0.0;
        }
    }
}
void matrixCopy(const double *source, double *dest, int row, int column)
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < column; j++)
        {
            dest[ i*column + j ] = source[ i*column + j];
        }
    }
}
void matrixPrint(const double *a, int row, int column)
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < column; j++)
        {
            std::cout << a[ i*column + j] << " ";
        }
        std::cout << std::endl;
    }
}
void matrixAdd(const double *a, const double *b, double *c, int row, int column)
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < column; j++)
        {
            c[i*column+j] = a[i*column+j] + b[i*column+j];
        }
    }
}
void matrixSubtract(const double *a, const double *b, double *c, int row, int column)
{
   for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < column; j++)
        {
            c[i*column+j] = a[i*column+j] - b[i*column+j];
        }
    } 
}
void matrixScalarMultiply(double *a, int row, int column, double scalar)
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < column; j++)
        {
            a[i*column+j] = a[i*column+j] * scalar;
        }
    }
}
bool matrixMultiply(const double *a, const double *b, int rowA, int colA, int rowB, int colB, double *c)
{
    if(colA != rowB) return false;
    if(a == nullptr || b == nullptr || c== nullptr) return false;
    for(int i = 0; i < rowA; i++)
    {
        for(int j = 0; j < colB; j++)
        {
            c[i*colB+j] = 0.0;
            for(int k = 0; k < colA; k++)
            {
                c[i*colB+j] += a[i*colA+k] * b[k*colB+j];
            }
        }
    }
    return true;
}
void matrixTranspose(const double *a, int row, int column, double *at)
{
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < column; j++)
        {
           at[j*row + i] = a[i*column + j];
        }
    }
}
void matrixRowSum(const double *a, double *result, int row, int column)
{
    for(int i = 0; i < row; i++)
    {
        double rowSum = 0.0;
        for(int j = 0; j < column; j++)
        {
            rowSum += a[i*column+j];
        }
        result[i] = rowSum;
    }
}
void matrixRowMax(const double *a, double *result, int row, int column)
{
   for(int i = 0; i < row; i++)
    {
        double maxValue = a[i*column];
        for(int j = 0; j < column; j++)
        {
            double value = a[i*column + j];
            if(maxValue < value) maxValue = value;
        }
    result[i] = maxValue;   
    } 
}
void matrixExpo(double *a, int row, int column)
{
   for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < column; j++)
        {
           a[i*column+j] = advanced_math::exponential(a[i*column+j]);
        }
    } 
}
void matrixLog(double *a, int row, int column)
{
   for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < column; j++)
        {
           a[i*column+j] = advanced_math::logarithm(a[i*column+j]);
        }
    } 
}
void matrixAddRowVec(double *a, const double *v, int row, int column)
{
  for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < column; j++)
        {
           a[i*column+j] += v[j];
        }
    }   
}
void matrixDivRowVec(double *a, const double *v, int row, int column)
{
  for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < column; j++)
        {
           a[i*column+j] /= v[j];
        }
    }   
}
}