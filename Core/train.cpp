#include<iostream>
#include "lstm.h"
#include "dense.h"
#include "cross_entropy.h"
#include "advanced_math.h"
#include "rng.h"
void bpttTruncated(const lstmState& state, const double *cellPrev, const double *gradHidden, double *forgetWeight, double *forgetBias, double *inputWeight,
                   double *inputBias, double *outputWeight, double *outputBias, double *candidateWeight, double *candidateBias, int inputSize, 
                   int hiddenSize, double learningRate)
{
    int concatSize = inputSize + hiddenSize;
}
int main()
{

}