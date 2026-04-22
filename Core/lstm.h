#ifndef LSTM_H
#define LSTM_H
#include "advanced_math.h"

struct lstmState
{
    double* hidden;
    double* cell;
    double* forget;
    double* inputGate;
    double* outputGate;
    double* candidate;
    double* concat;
};

void initLstmState(lstmState& s, int hiddenSize, int concatSize);
void freeLstmState(lstmState& s);
void concatInput(const double* input, int inputSize, const double* hiddenPrev, int hiddenSize, double* concat);
void denseForward(const double* input, const double* weight, const double* bias,double* output, int inputSize, int outputSize);
void lstmForward(const double* input, const double* hiddenPrev, const double* cellPrev,const double* forgetGateWeight, const double* forgetGateBias,
                 const double* inputGateWeight, const double* inputGateBias,const double* outputGateWeight, const double* outputGateBias,
                 const double* candidateGateWeight, const double* candidateGateBias,int inputSize, int hiddenSize, lstmState& state);

#endif