#include"dense.h"
#include"rng.h"

dense::denseLayer::denseLayer(int input, int output)
{
    inputSize = input;
    outputSize = output;
    weight = new double[input * output];
    bias = new double[output];
    inputCache = new double[input];
    for(int i = 0; i < input*output; i++)
    {
        weight[i] = rng::uniform(-0.05, 0.05); 
    }
    for(int i = 0; i < output; i++)
    {
        bias[i] = 0.0;
    }
}
dense::denseLayer::~denseLayer()
{
    delete[] weight;
    delete[] bias;
    delete[] inputCache;
}
void dense::denseLayer::forward(const double *input, double *output)
{
    for (int i = 0; i < inputSize; i++)
    {
        inputCache[i] = input[i];
    } 
    for (int j = 0; j < outputSize; j++)
    {
       double sum = bias[j];
       for(int i = 0; i < inputSize; i++)
       {
         sum += weight[j * inputSize + i] * input[i];
       }
       output[j] = sum;
    }
}
void dense::denseLayer::backward(const double *denseOutput, double *denseInput, double learningRate)
{
    for(int i = 0; i < inputSize; i++)
    {
        denseInput[i] = 0.0;
    }

    for(int j = 0; j < outputSize; j++)
    {
        for(int i = 0; i < inputSize; i++)
        {
            int index = j * inputSize + i;
            denseInput[i] += weight[index] * denseOutput[j];
            double denseWeight = denseOutput[j] * inputCache[i];
            weight[index] -= learningRate * denseWeight;
        }
        bias[j] -= learningRate * denseOutput[j];
    }
}
