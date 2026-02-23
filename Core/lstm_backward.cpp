#include "lstm_backward.h"
#include "advanced_math.h"
#include <vector>

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

    static std::vector<double> gradForgetW;
    static std::vector<double> gradInputW;
    static std::vector<double> gradOutputW;
    static std::vector<double> gradCandW;

    static std::vector<double> gradForgetB;
    static std::vector<double> gradInputB;
    static std::vector<double> gradOutputB;
    static std::vector<double> gradCandB;

    static std::vector<double> dHiddenNext;
    static std::vector<double> dCellNext;

    static std::vector<double> dHiddenPrev;
    static std::vector<double> dCellPrev;
    static std::vector<double> dConcat;

    const int wSize = hiddenSize * concatSize;

    if((int)gradForgetW.size() != wSize)
    {
        gradForgetW.assign(wSize, 0.0);
        gradInputW.assign(wSize, 0.0);
        gradOutputW.assign(wSize, 0.0);
        gradCandW.assign(wSize, 0.0);
    }
    else
    {
        for(int i = 0; i < wSize; i++)
        {
            gradForgetW[i] = 0.0;
            gradInputW[i]  = 0.0;
            gradOutputW[i] = 0.0;
            gradCandW[i]   = 0.0;
        }
    }

    if((int)gradForgetB.size() != hiddenSize)
    {
        gradForgetB.assign(hiddenSize, 0.0);
        gradInputB.assign(hiddenSize, 0.0);
        gradOutputB.assign(hiddenSize, 0.0);
        gradCandB.assign(hiddenSize, 0.0);
    }
    else
    {
        for(int i = 0; i < hiddenSize; i++)
        {
            gradForgetB[i] = 0.0;
            gradInputB[i]  = 0.0;
            gradOutputB[i] = 0.0;
            gradCandB[i]   = 0.0;
        }
    }

    if((int)dHiddenNext.size() != hiddenSize)
    {
        dHiddenNext.assign(hiddenSize, 0.0);
        dCellNext.assign(hiddenSize, 0.0);
        dHiddenPrev.assign(hiddenSize, 0.0);
        dCellPrev.assign(hiddenSize, 0.0);
    }
    else
    {
        for(int i = 0; i < hiddenSize; i++)
        {
            dHiddenNext[i] = 0.0;
            dCellNext[i]   = 0.0;
            dHiddenPrev[i] = 0.0;
            dCellPrev[i]   = 0.0;
        }
    }

    if((int)dConcat.size() != concatSize)
        dConcat.assign(concatSize, 0.0);
    else
        for(int j = 0; j < concatSize; j++) dConcat[j] = 0.0;

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

        for(int i = 0; i < hiddenSize; i++)
        {
            dHiddenPrev[i] = 0.0;
            dCellPrev[i] = 0.0;
        }
        for(int j = 0; j < concatSize; j++)
            dConcat[j] = 0.0;

        for(int i = 0; i < hiddenSize; i++)
        {
            const double forgetGateVal = st.forget[i];
            const double inputGateVal  = st.inputGate[i];
            const double outputGateVal = st.outputGate[i];
            const double candidateVal  = st.candidate[i];

            const double cellVal = st.cell[i];
            const double tanhCell = advanced_math::tanh(cellVal);

            const double dHidden = dHiddenNext[i];

            double dOutput = dHidden * tanhCell;

            double dC = dHidden * outputGateVal * (1.0 - tanhCell * tanhCell);
            dC += dCellNext[i];

            double dForget = dC * cPrev[i];
            double dInput  = dC * candidateVal;
            double dCand   = dC * inputGateVal;

            double dO_pre = dOutput * outputGateVal * (1.0 - outputGateVal);
            double dF_pre = dForget * forgetGateVal * (1.0 - forgetGateVal);
            double dI_pre = dInput  * inputGateVal  * (1.0 - inputGateVal);
            double dG_pre = dCand   * (1.0 - candidateVal * candidateVal);

            dO_pre = clip(dO_pre, gradClip);
            dF_pre = clip(dF_pre, gradClip);
            dI_pre = clip(dI_pre, gradClip);
            dG_pre = clip(dG_pre, gradClip);

            for(int j = 0; j < concatSize; j++)
            {
                const double concatVal = st.concat[j];
                const int idx = i * concatSize + j;

                gradForgetW[idx] += dF_pre * concatVal;
                gradInputW[idx]  += dI_pre * concatVal;
                gradOutputW[idx] += dO_pre * concatVal;
                gradCandW[idx]   += dG_pre * concatVal;

                dConcat[j] += forgetGateWeight[idx] * dF_pre
                           +  inputGateWeight[idx] * dI_pre
                           + outputGateWeight[idx] * dO_pre
                           +   candidateWeight[idx] * dG_pre;
            }

            gradForgetB[i] += dF_pre;
            gradInputB[i]  += dI_pre;
            gradOutputB[i] += dO_pre;
            gradCandB[i]   += dG_pre;

            dCellPrev[i] = dC * forgetGateVal;
        }

        for(int j = 0; j < hiddenSize; j++)
            dHiddenPrev[j] = dConcat[inputSize + j];

        for(int i = 0; i < hiddenSize; i++)
        {
            dHiddenNext[i] = dHiddenPrev[i];
            dCellNext[i]   = dCellPrev[i];
        }
    }

    double scale = 1.0;
    if(k > 0) scale = 1.0 / (double)k;

    const double lr = learningRate * scale;

    for(int i = 0; i < wSize; i++)
    {
        forgetGateWeight[i] -= lr * gradForgetW[i];
        inputGateWeight[i]  -= lr * gradInputW[i];
        outputGateWeight[i] -= lr * gradOutputW[i];
        candidateWeight[i]  -= lr * gradCandW[i];
    }

    for(int i = 0; i < hiddenSize; i++)
    {
        forgetGateBias[i] -= lr * gradForgetB[i];
        inputGateBias[i]  -= lr * gradInputB[i];
        outputGateBias[i] -= lr * gradOutputB[i];
        candidateBias[i]  -= lr * gradCandB[i];
    }
}
