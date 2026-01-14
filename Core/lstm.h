#ifndef LSTM_H
#define LSTM_H
#include "advanced_math.h"
struct lstmState
{
  double *hidden;
  double *cell;
  double *forget;
  double *inputGate;
  double *outputGate;
  double *candidate;
  double *concat; 
};
void initLstmState(lstmState& s, int hiddenSize, int concatSize);
void freeLstmState(lstmState& s);
void concatInput(const double *x, int inputSize, const double *hPrev, int hiddenSize, double *concat);
void denseForward(const double *input, const double *weight, const double *bias, double *output, int inputSize, int outputSize);
void lstmForward(const double *x, const double *hPrev, const double* cPrev, const double *fGateWeight, const double *fGateBias,
    const double *iGateWeight, const double *iGateBias, const double *oGateWeight, const double *oGateBias,
    const double *canGateWeight, const double *canGateBias, int inputSize, int hiddenSize, lstmState& state);
#endif