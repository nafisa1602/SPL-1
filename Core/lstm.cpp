#include "lstm.h"
#include "advanced_math.h"
#include "matrix_math.h"

void initLstmState(lstmState& s, int hiddenSize, int concatSize)
{
    s.hidden = new double[hiddenSize];
    s.cell = new double[hiddenSize];
    s.forget = new double[hiddenSize];
    s.inputGate = new double[hiddenSize];
    s.outputGate = new double[hiddenSize];
    s.candidate = new double[hiddenSize];
    s.concat = new double[concatSize];
    matrix_math::matrixZero(s.hidden, 1, hiddenSize);
    matrix_math::matrixZero(s.cell, 1, hiddenSize);
    matrix_math::matrixZero(s.forget, 1, hiddenSize);
    matrix_math::matrixZero(s.inputGate, 1, hiddenSize);
    matrix_math::matrixZero(s.outputGate, 1, hiddenSize);
    matrix_math::matrixZero(s.candidate, 1, hiddenSize);
    matrix_math::matrixZero(s.concat, 1, concatSize);
}

void freeLstmState(lstmState& s)
{
    delete[] s.hidden;
    delete[] s.cell;
    delete[] s.forget;
    delete[] s.inputGate;
    delete[] s.outputGate;
    delete[] s.candidate;
    delete[] s.concat;
}

void concatInput(const double* input, int inputSize, const double* hiddenPrev, int hiddenSize, double* concat)
{
    for (int i = 0; i < inputSize; i++) concat[i] = input[i];
    for (int i = 0; i < hiddenSize; i++) concat[inputSize + i] = hiddenPrev[i];
}

void denseForward(const double* input, const double* weight, const double* bias,double* output, int inputSize, int outputSize)
{
    for (int outIdx = 0; outIdx < outputSize; outIdx++) 
    {
        double sum = bias[outIdx];
        for (int inIdx = 0; inIdx < inputSize; inIdx++) 
        {
            sum += weight[outIdx * inputSize + inIdx] * input[inIdx];
        }
        output[outIdx] = sum;
    }
}

void lstmForward(const double* input, const double* hiddenPrev, const double* cellPrev, const double* fGateWeight, const double* fGateBias,
    const double* iGateWeight,   const double* iGateBias, const double* oGateWeight,   const double* oGateBias,
    const double* canGateWeight, const double* canGateBias, int inputSize, int hiddenSize, lstmState& state)
{
    const int concatSize = inputSize + hiddenSize;
    concatInput(input, inputSize, hiddenPrev, hiddenSize, state.concat);
    denseForward(state.concat, fGateWeight,   fGateBias,   state.forget,     concatSize, hiddenSize);
    denseForward(state.concat, iGateWeight,   iGateBias,   state.inputGate,  concatSize, hiddenSize);
    denseForward(state.concat, oGateWeight,   oGateBias,   state.outputGate, concatSize, hiddenSize);
    denseForward(state.concat, canGateWeight, canGateBias, state.candidate,  concatSize, hiddenSize);
    for (int i = 0; i < hiddenSize; i++) 
    {
        state.forget[i]    = advanced_math::sigmoid(state.forget[i]);
        state.inputGate[i] = advanced_math::sigmoid(state.inputGate[i]);
        state.outputGate[i] = advanced_math::sigmoid(state.outputGate[i]);
        state.candidate[i] = advanced_math::tanh(state.candidate[i]);
    }
    for (int i = 0; i < hiddenSize; i++) 
    {
        double cell = state.forget[i] * cellPrev[i] + state.inputGate[i] * state.candidate[i];
        // Clamp cell state to prevent unbounded growth across thousands of timesteps.
        // tanh is already bounded to [-1,1] so hidden stays bounded too.
        state.cell[i] = advanced_math::clamp(cell, -10.0, 10.0);
    }
    for (int i = 0; i < hiddenSize; i++) 
    {
        state.hidden[i] = state.outputGate[i] * advanced_math::tanh(state.cell[i]);
    }
}