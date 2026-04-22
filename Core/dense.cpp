#include "dense.h"
#include "rng.h"
#include "advanced_math.h"
#include "matrix_math.h"

dense::denseLayer::denseLayer(int input, int output)
{
    inputSize = input;
    outputSize = output;
    weight = new double[input * output];
    bias = new double[output];
    inputCache = new double[input];
    double stddev = advanced_math::squareRoot(2.0 / (double)(input + output));
    for (int i = 0; i < input * output; i++) 
    {
        weight[i] = rng::uniform(-stddev, stddev);
    }
    matrix_math::matrixZero(bias, 1, output);
}

dense::denseLayer::~denseLayer()
{
    delete[] weight;
    delete[] bias;
    delete[] inputCache;
}

void dense::denseLayer::forward(const double* input, double* output)
{
    matrix_math::matrixCopy(input, inputCache, 1, inputSize);
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

void dense::denseLayer::backward(const double* denseOutput, double* denseInput, double learningRate)
{
    const double gradClip = 1.0;
    matrix_math::matrixZero(denseInput, 1, inputSize);
    for (int outIdx = 0; outIdx < outputSize; outIdx++) 
    {
        double outputGrad = denseOutput[outIdx];
        for (int inIdx = 0; inIdx < inputSize; inIdx++) 
        {
            int index = outIdx * inputSize + inIdx;
            // Clip gradient flowing back to LSTM
            double clippedOutputGrad = advanced_math::clamp(outputGrad, -gradClip, gradClip);
            denseInput[inIdx] += weight[index] * clippedOutputGrad;
            // Keep forward signal intact, then clip update magnitude.
            double weightGrad = advanced_math::clamp(outputGrad * inputCache[inIdx], -gradClip, gradClip);
            weight[index] -= learningRate * weightGrad;
        }
        // Bias gradient: clip before applying
        double biasGrad = advanced_math::clamp(denseOutput[outIdx], -gradClip, gradClip);
        bias[outIdx] -= learningRate * biasGrad;
    }
    // Clip input gradient flowing into LSTM
    for (int i = 0; i < inputSize; i++) 
    {
        denseInput[i] = advanced_math::clamp(denseInput[i], -gradClip, gradClip);
    }
}