#include "lstm_backward.h"
#include "advanced_math.h"
#include <vector>

static inline double clip(double x, double limit)
{
    if(limit <= 0.0) return x;
    if(x >  limit) return  limit;
    if(x < -limit) return -limit;
    return x;
}

static double computeGradientNorm(
    const std::vector<double>& gFW, const std::vector<double>& gIW,
    const std::vector<double>& gOW, const std::vector<double>& gCW,
    const std::vector<double>& gFB, const std::vector<double>& gIB,
    const std::vector<double>& gOB, const std::vector<double>& gCB)
{
    double norm_sq = 0.0;
    for(size_t i = 0; i < gFW.size(); i++) norm_sq += gFW[i] * gFW[i];
    for(size_t i = 0; i < gIW.size(); i++) norm_sq += gIW[i] * gIW[i];
    for(size_t i = 0; i < gOW.size(); i++) norm_sq += gOW[i] * gOW[i];
    for(size_t i = 0; i < gCW.size(); i++) norm_sq += gCW[i] * gCW[i];
    for(size_t i = 0; i < gFB.size(); i++) norm_sq += gFB[i] * gFB[i];
    for(size_t i = 0; i < gIB.size(); i++) norm_sq += gIB[i] * gIB[i];
    for(size_t i = 0; i < gOB.size(); i++) norm_sq += gOB[i] * gOB[i];
    for(size_t i = 0; i < gCB.size(); i++) norm_sq += gCB[i] * gCB[i];
    return advanced_math::squareRoot(norm_sq);
}

void lstmBackwardTruncated(
    const lstmState *state, int t, int k, const double *cZero, const double *dHiddenT,
    double *forgetGateWeight, double *forgetGateBias,
    double *inputGateWeight,  double *inputGateBias,
    double *outputGateWeight, double *outputGateBias,
    double *candidateWeight,  double *candidateBias,
    int inputSize, int hiddenSize,
    double learningRate, double gradClip
)
{
    const int concatSize = inputSize + hiddenSize;
    const int wSize = hiddenSize * concatSize;

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

    // Resize or zero all gradient accumulators
    if((int)gradForgetW.size() != wSize)
    {
        gradForgetW.assign(wSize, 0.0); gradInputW.assign(wSize, 0.0);
        gradOutputW.assign(wSize, 0.0); gradCandW.assign(wSize, 0.0);
    }
    else
    {
        for(int i = 0; i < wSize; i++)
        { gradForgetW[i]=0.0; gradInputW[i]=0.0; gradOutputW[i]=0.0; gradCandW[i]=0.0; }
    }

    if((int)gradForgetB.size() != hiddenSize)
    {
        gradForgetB.assign(hiddenSize, 0.0); gradInputB.assign(hiddenSize, 0.0);
        gradOutputB.assign(hiddenSize, 0.0); gradCandB.assign(hiddenSize, 0.0);
    }
    else
    {
        for(int i = 0; i < hiddenSize; i++)
        { gradForgetB[i]=0.0; gradInputB[i]=0.0; gradOutputB[i]=0.0; gradCandB[i]=0.0; }
    }

    if((int)dHiddenNext.size() != hiddenSize)
    {
        dHiddenNext.assign(hiddenSize, 0.0); dCellNext.assign(hiddenSize, 0.0);
        dHiddenPrev.assign(hiddenSize, 0.0); dCellPrev.assign(hiddenSize, 0.0);
    }
    else
    {
        for(int i = 0; i < hiddenSize; i++)
        { dHiddenNext[i]=0.0; dCellNext[i]=0.0; dHiddenPrev[i]=0.0; dCellPrev[i]=0.0; }
    }

    if((int)dConcat.size() != concatSize)
        dConcat.assign(concatSize, 0.0);
    else
        for(int j = 0; j < concatSize; j++) dConcat[j] = 0.0;

    // Clip incoming gradient from classifier before entering BPTT
    for(int i = 0; i < hiddenSize; i++)
    {
        dHiddenNext[i] = clip(dHiddenT[i], gradClip);
        dCellNext[i]   = 0.0;
    }

    int tEnd = t - k;
    if(tEnd < 0) tEnd = 0;

    for(int time = t - 1; time >= tEnd; time--)
    {
        const lstmState& st = state[time];
        const double *cPrev = (time == 0) ? cZero : state[time - 1].cell;

        for(int i = 0; i < hiddenSize; i++) { dHiddenPrev[i]=0.0; dCellPrev[i]=0.0; }
        for(int j = 0; j < concatSize; j++) dConcat[j] = 0.0;

        for(int i = 0; i < hiddenSize; i++)
        {
            const double forgetGateVal = st.forget[i];
            const double inputGateVal  = st.inputGate[i];
            const double outputGateVal = st.outputGate[i];
            const double candidateVal  = st.candidate[i];
            const double tanhCell      = advanced_math::tanh(st.cell[i]);
            const double dHidden       = dHiddenNext[i];

            double dOutput = dHidden * tanhCell;

            // Cell gradient: do NOT clip here; let global norm clipping handle it
            double dC = dHidden * outputGateVal * (1.0 - tanhCell * tanhCell);
            dC += dCellNext[i];

            double dForget = dC * cPrev[i];
            double dInput  = dC * candidateVal;
            double dCand   = dC * inputGateVal;

            // Gate pre-activation gradients: NO intermediate clipping
            // Let these flow naturally; global norm clipping at end handles explosion
            double dO_pre = dOutput * outputGateVal * (1.0 - outputGateVal);
            double dF_pre = dForget * forgetGateVal * (1.0 - forgetGateVal);
            double dI_pre = dInput  * inputGateVal  * (1.0 - inputGateVal);
            double dG_pre = dCand   * (1.0 - candidateVal * candidateVal);

            for(int j = 0; j < concatSize; j++)
            {
                const double concatVal = st.concat[j];
                const int idx = i * concatSize + j;

                gradForgetW[idx] += dF_pre * concatVal;
                gradInputW[idx]  += dI_pre * concatVal;
                gradOutputW[idx] += dO_pre * concatVal;
                gradCandW[idx]   += dG_pre * concatVal;

                dConcat[j] += forgetGateWeight[idx] * dF_pre
                            + inputGateWeight[idx]  * dI_pre
                            + outputGateWeight[idx] * dO_pre
                            + candidateWeight[idx]  * dG_pre;
            }

            gradForgetB[i] += dF_pre;
            gradInputB[i]  += dI_pre;
            gradOutputB[i] += dO_pre;
            gradCandB[i]   += dG_pre;

            dCellPrev[i] = dC * forgetGateVal;
        }

        // Extract dHiddenPrev from dConcat (do NOT clip dConcat anymore)
        for(int j = 0; j < hiddenSize; j++)
            dHiddenPrev[j] = dConcat[inputSize + j];

        for(int i = 0; i < hiddenSize; i++)
        {
            dHiddenNext[i] = dHiddenPrev[i];
            dCellNext[i]   = dCellPrev[i];
        }
    }

    // Average gradients over k timesteps
    double invK = (k > 0) ? 1.0 / (double)k : 1.0;
    for(int i = 0; i < wSize; i++)
    {
        gradForgetW[i] *= invK; gradInputW[i]  *= invK;
        gradOutputW[i] *= invK; gradCandW[i]   *= invK;
    }
    for(int i = 0; i < hiddenSize; i++)
    {
        gradForgetB[i] *= invK; gradInputB[i]  *= invK;
        gradOutputB[i] *= invK; gradCandB[i]   *= invK;
    }

    // Global gradient norm clipping
    double grad_norm = computeGradientNorm(
        gradForgetW, gradInputW, gradOutputW, gradCandW,
        gradForgetB, gradInputB, gradOutputB, gradCandB);

    double clip_scale = (grad_norm > gradClip) ? gradClip / grad_norm : 1.0;
    const double lr = learningRate;

    for(int i = 0; i < wSize; i++)
    {
        forgetGateWeight[i] -= lr * clip_scale * gradForgetW[i];
        inputGateWeight[i]  -= lr * clip_scale * gradInputW[i];
        outputGateWeight[i] -= lr * clip_scale * gradOutputW[i];
        candidateWeight[i]  -= lr * clip_scale * gradCandW[i];
    }

    for(int i = 0; i < hiddenSize; i++)
    {
        forgetGateBias[i] -= lr * clip_scale * gradForgetB[i];
        inputGateBias[i]  -= lr * clip_scale * gradInputB[i];
        outputGateBias[i] -= lr * clip_scale * gradOutputB[i];
        candidateBias[i]  -= lr * clip_scale * gradCandB[i];
    }
}