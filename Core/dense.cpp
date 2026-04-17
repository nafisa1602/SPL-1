#include "dense.h"
#include "rng.h"
#include "advanced_math.h"

dense::denseLayer::denseLayer(int input, int output)
{
    inputSize = input;
    outputSize = output;
    weight = new double[input * output];
    bias = new double[output];
    inputCache = new double[input];

    double stddev = advanced_math::squareRoot(2.0 / (double)(input + output));
    for (int i = 0; i < input * output; i++) {
        weight[i] = rng::uniform(-stddev, stddev);
    }
    for (int i = 0; i < output; i++) {
        bias[i] = 0.0;
    }
}

dense::denseLayer::~denseLayer()
{
    delete[] weight;
    delete[] bias;
    delete[] inputCache;
}

void dense::denseLayer::forward(const double* input, double* output)
{
    for (int i = 0; i < inputSize; i++) {
        inputCache[i] = input[i];
    }

    for (int outIdx = 0; outIdx < outputSize; outIdx++) {
        double sum = bias[outIdx];
        for (int inIdx = 0; inIdx < inputSize; inIdx++) {
            sum += weight[outIdx * inputSize + inIdx] * input[inIdx];
        }
        output[outIdx] = sum;
    }
}

void dense::denseLayer::backward(const double* denseOutput, double* denseInput, double learningRate)
{
    const double gradClip = 1.0;

    for (int i = 0; i < inputSize; i++) {
        denseInput[i] = 0.0;
    }

    for (int outIdx = 0; outIdx < outputSize; outIdx++) {
        double outputGrad = denseOutput[outIdx];

        for (int inIdx = 0; inIdx < inputSize; inIdx++) {
            int index = outIdx * inputSize + inIdx;

            // Clip gradient flowing back to LSTM
            double clippedOutputGrad = outputGrad;
            if (clippedOutputGrad > gradClip) clippedOutputGrad = gradClip;
            if (clippedOutputGrad < -gradClip) clippedOutputGrad = -gradClip;

            denseInput[inIdx] += weight[index] * clippedOutputGrad;

            // Keep forward signal intact, then clip update magnitude.
            double weightGrad = outputGrad * inputCache[inIdx];
            if (weightGrad > gradClip) weightGrad = gradClip;
            if (weightGrad < -gradClip) weightGrad = -gradClip;

            weight[index] -= learningRate * weightGrad;
        }

        // Bias gradient: clip before applying
        double biasGrad = denseOutput[outIdx];
        if (biasGrad > gradClip) biasGrad = gradClip;
        if (biasGrad < -gradClip) biasGrad = -gradClip;
        bias[outIdx] -= learningRate * biasGrad;
    }

    // Clip input gradient flowing into LSTM
    for (int i = 0; i < inputSize; i++) {
        if (denseInput[i] > gradClip) denseInput[i] = gradClip;
        if (denseInput[i] < -gradClip) denseInput[i] = -gradClip;
    }
}