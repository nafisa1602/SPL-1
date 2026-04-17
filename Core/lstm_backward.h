#ifndef LSTM_BACKWARD_H
#define LSTM_BACKWARD_H

#include <vector>
#include "lstm.h"

void lstmBackwardTruncated(
    const lstmState* state, int timeSteps, int truncationWindow,
    const double* cellZero, const double* hiddenGradAtT,
    double* forgetGateWeight, double* forgetGateBias,
    double* inputGateWeight, double* inputGateBias,
    double* outputGateWeight, double* outputGateBias,
    double* candidateWeight, double* candidateBias,
    int inputSize, int hiddenSize,
    double learningRate, double gradClip
);

#endif