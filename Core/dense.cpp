#include "dense.h"
#include "rng.h"
#include "advanced_math.h"

dense::denseLayer::denseLayer(int input, int output)
{
    inputSize  = input;
    outputSize = output;
    weight     = new double[input * output];
    bias       = new double[output];
    inputCache = new double[input];
    double stddev = advanced_math::squareRoot(2.0 / (double)(input + output));
    for(int i = 0; i < input * output; i++)
        weight[i] = rng::uniform(-stddev, stddev);
    for(int i = 0; i < output; i++)
        bias[i] = 0.0;
}

dense::denseLayer::~denseLayer()
{
    delete[] weight;
    delete[] bias;
    delete[] inputCache;
}

void dense::denseLayer::forward(const double *input, double *output)
{
    for(int i = 0; i < inputSize; i++)
        inputCache[i] = input[i];
    for(int j = 0; j < outputSize; j++)
    {
        double sum = bias[j];
        for(int i = 0; i < inputSize; i++)
            sum += weight[j * inputSize + i] * input[i];
        output[j] = sum;
    }
}

void dense::denseLayer::backward(const double *denseOutput, double *denseInput, double learningRate)
{
    const double gradClip = 1.0;

    for(int i = 0; i < inputSize; i++)
        denseInput[i] = 0.0;

    for(int j = 0; j < outputSize; j++)
    {
        double dOut = denseOutput[j];

        for(int i = 0; i < inputSize; i++)
        {
            int index = j * inputSize + i;
            
            // Clip gradient flowing back to LSTM
            double dOutClipped = dOut;
            if(dOutClipped >  gradClip) dOutClipped =  gradClip;
            if(dOutClipped < -gradClip) dOutClipped = -gradClip;
            
            denseInput[i] += weight[index] * dOutClipped;

            // Compute weight gradient WITHOUT pre-clipping dOut
            double dW = dOut * inputCache[i];
            if(dW >  gradClip) dW =  gradClip;
            if(dW < -gradClip) dW = -gradClip;

            weight[index] -= learningRate * dW;
        }

        // Bias gradient: clip before applying
        double dB = denseOutput[j];
        if(dB >  gradClip) dB =  gradClip;
        if(dB < -gradClip) dB = -gradClip;
        bias[j] -= learningRate * dB;
    }

    // Clip input gradient flowing into LSTM
    for(int i = 0; i < inputSize; i++)
    {
        if(denseInput[i] >  gradClip) denseInput[i] =  gradClip;
        if(denseInput[i] < -gradClip) denseInput[i] = -gradClip;
    }
}