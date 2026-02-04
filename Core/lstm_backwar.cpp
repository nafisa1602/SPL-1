#include "lstm_backward.h"
#include "advanced_math.h"
static inline double clip(double x, double limit)
{
    if(limit <= 0.0) return x;
    if(x > limit) return limit;
    if(x < -limit) return -limit;
    return x;
}
void lstmBackwardTruncated
(
    const lstmState *state, int t, int k, const double *cZero, const double *dHiddenT, double *forgetGateWeight,
    double *forgetGateBias, double *inputGateWeight, double *inputGateBias, double *outputGateWeight,
    double *outputGateBias, double *candidateWeight, double *candidateBias, int inputSize, int hiddenSize,
    double learningRate, double gradClip
)
{
    const int concatSize = inputSize + hiddenSize;

    double *dHiddenNext = new double[hiddenSize];
    double *dCellNext = new double[hiddenSize];

    for(int i = 0; i < hiddenSize; i++)
    {
        dHiddenNext[i] = dHiddenT[i];
        dCellNext[i] = 0.0;
    }

    int tEnd = t - k;
    if(tEnd < 0) tEnd = 0;
    for(int time = t - 1; time >= tEnd; time--)
    {
        const lstmState& st = state[time];
        const double *cPrev = (time == 0) ? cZero : state[time - 1].cell;

        double *dHiddenPrev = new double[hiddenSize];
        double *dCellPrev = new double[hiddenSize];

        for(int i = 0; i < hiddenSize; i++)
        {
            dHiddenPrev[i] = 0.0;
            dCellPrev[i] = 0.0;
        }

        for(int i = 0; i < hiddenSize; i++)
        {
            const double forgetGateVal = st.forget[i];
            const double inputGateVal = st.inputGate[i];
            const double outputGateVal = st.outputGate[i];
            const double candidateVal = st.candidate[i];

            const double cellVal = st.cell[i];
            const double tanhCell = advanced_math::tanh(cellVal);

            const double dHidden = dHiddenNext[i];

            double dOutput = dHidden * tanhCell;

            double dC = dHidden * outputGateVal * (1.0 - tanhCell * tanhCell);
            dC += dCellNext[i];

            double dForget = dC * cPrev[i];
            double dInput = dC * candidateVal;
            double dCandidate = dC * inputGateVal;

            double dO_pre = dOutput * outputGateVal * (1.0 - outputGateVal);
            double dF_pre = dForget * forgetGateVal * (1.0 - forgetGateVal);
            double dI_pre = dInput * inputGateVal * (1.0 - inputGateVal);
            double dG_pre = dCandidate * (1.0 - candidateVal * candidateVal);

            dO_pre = clip(dO_pre, gradClip);
            dF_pre = clip(dF_pre, gradClip);
            dI_pre = clip(dI_pre, gradClip);
            dG_pre = clip(dG_pre, gradClip);

            for(int j = 0; j < concatSize; j++)
            {
                const double concatVal = st.concat[j];
                const int idx = i * concatSize + j;

                forgetGateWeight[idx] -= learningRate * dF_pre * concatVal;
                inputGateWeight[idx] -= learningRate * dI_pre * concatVal;
                outputGateWeight[idx] -= learningRate * dO_pre * concatVal;
                candidateWeight[idx] -= learningRate * dG_pre * concatVal;
            }

            forgetGateBias[i] -= learningRate * dF_pre;
            inputGateBias[i] -= learningRate * dI_pre;
            outputGateBias[i] -= learningRate * dO_pre;
            candidateBias[i] -= learningRate * dG_pre;

            dCellPrev[i] = dC * forgetGateVal;

            double dhprev = 0.0;
            for(int j = 0; j < hiddenSize; j++)
            {
                const int col = inputSize + j;
                const int widx = i * concatSize + col;

                dhprev += forgetGateWeight[widx] * dF_pre
                       +  inputGateWeight[widx] * dI_pre
                       + outputGateWeight[widx] * dO_pre
                       +   candidateWeight[widx] * dG_pre;
            }

            dHiddenPrev[i] += dhprev;
        }

        for(int i = 0; i < hiddenSize; i++)
        {
            dHiddenNext[i] = dHiddenPrev[i];
            dCellNext[i] = dCellPrev[i];
        }

        delete[] dHiddenPrev;
        delete[] dCellPrev;
    }

    delete[] dHiddenNext;
    delete[] dCellNext;
}
