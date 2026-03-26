#include "lstm.h"
#include "advanced_math.h"

void initLstmState(lstmState& s, int hiddenSize, int concatSize)
{
    s.hidden    = new double[hiddenSize];
    s.cell      = new double[hiddenSize];
    s.forget    = new double[hiddenSize];
    s.inputGate = new double[hiddenSize];
    s.outputGate= new double[hiddenSize];
    s.candidate = new double[hiddenSize];
    s.concat    = new double[concatSize];
    for(int i = 0; i < hiddenSize; i++)
    {
        s.hidden[i] = 0.0; s.cell[i]      = 0.0;
        s.forget[i] = 0.0; s.inputGate[i] = 0.0;
        s.outputGate[i] = 0.0; s.candidate[i] = 0.0;
    }
    for(int i = 0; i < concatSize; i++) s.concat[i] = 0.0;
}

void freeLstmState(lstmState& s)
{
    delete[] s.hidden; delete[] s.cell;    delete[] s.forget;
    delete[] s.inputGate; delete[] s.outputGate; delete[] s.candidate;
    delete[] s.concat;
}

void concatInput(const double *x, int inputSize, const double *hPrev, int hiddenSize, double *concat)
{
    for(int i = 0; i < inputSize;  i++) concat[i]            = x[i];
    for(int i = 0; i < hiddenSize; i++) concat[inputSize + i] = hPrev[i];
}

void denseForward(const double *input, const double *weight, const double *bias,
                  double *output, int inputSize, int outputSize)
{
    for(int i = 0; i < outputSize; i++)
    {
        double sum = bias[i];
        for(int j = 0; j < inputSize; j++)
            sum += weight[i * inputSize + j] * input[j];
        output[i] = sum;
    }
}

void lstmForward(const double *x, const double *hPrev, const double *cPrev,
    const double *fGateWeight,   const double *fGateBias,
    const double *iGateWeight,   const double *iGateBias,
    const double *oGateWeight,   const double *oGateBias,
    const double *canGateWeight, const double *canGateBias,
    int inputSize, int hiddenSize, lstmState& state)
{
    int concatSize = inputSize + hiddenSize;
    concatInput(x, inputSize, hPrev, hiddenSize, state.concat);

    denseForward(state.concat, fGateWeight,   fGateBias,   state.forget,     concatSize, hiddenSize);
    denseForward(state.concat, iGateWeight,   iGateBias,   state.inputGate,  concatSize, hiddenSize);
    denseForward(state.concat, oGateWeight,   oGateBias,   state.outputGate, concatSize, hiddenSize);
    denseForward(state.concat, canGateWeight, canGateBias, state.candidate,  concatSize, hiddenSize);

    for(int i = 0; i < hiddenSize; i++)
    {
        state.forget[i]    = advanced_math::sigmoid(state.forget[i]);
        state.inputGate[i] = advanced_math::sigmoid(state.inputGate[i]);
        state.outputGate[i]= advanced_math::sigmoid(state.outputGate[i]);
        state.candidate[i] = advanced_math::tanh(state.candidate[i]);
    }

    for(int i = 0; i < hiddenSize; i++)
    {
        double cell = state.forget[i] * cPrev[i] + state.inputGate[i] * state.candidate[i];
        // Clamp cell state to prevent unbounded growth across thousands of timesteps.
        // tanh is already bounded to [-1,1] so hidden stays bounded too.
        state.cell[i] = advanced_math::clamp(cell, -10.0, 10.0);
    }

    for(int i = 0; i < hiddenSize; i++)
        state.hidden[i] = state.outputGate[i] * advanced_math::tanh(state.cell[i]);
}