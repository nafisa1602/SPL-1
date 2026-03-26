#ifndef LSTM_BACKWARD_H
#define LSTM_BACKWARD_H
#include <vector>
#include "lstm.h"

void lstmBackwardTruncated(
    const lstmState *state, int t, int k, const double *cZero, const double *dHiddenT,
    double *forgetGateWeight, double *forgetGateBias,
    double *inputGateWeight,  double *inputGateBias,
    double *outputGateWeight, double *outputGateBias,
    double *candidateWeight,  double *candidateBias,
    int inputSize, int hiddenSize,
    double learningRate, double gradClip
);

#endif