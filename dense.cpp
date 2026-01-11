#include"dense.h"
static unsigned int number = 123456;
static double randomNumber()
{
    number = number * 1103515245 + 12345;
    return ((number / 65536) % 32768) / 32768.0;
}
dense::denseLayer::denseLayer(int input, int output)
{
    inputSize = input;
    outputSize = output;
    weight = new double[input * output];
    bias = new double[output];
    inputCache = new double[input];
    for(int i = 0; i < input*output; i++)
    {
        weight[i] = randomNumber() * 0.1 - 0.05; //weight initialize korsi
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
    for (int i = 0; i < inputSize; i++)
    {
        inputCache[i] = denseInput[i];
    } 
    for(int j = 0; j < outputSize; j++)
    {
        for (int i = 0; i < inputSize; i++)
    {
        int index = j * inputSize + i;
        double denseWeight = denseOutput[j] * inputCache[i];
            denseInput[i] += weight[index] * denseOutput[j];

            weight[index] -= learningRate * denseWeight;
    } 
    bias[j] -= learningRate * denseOutput[j];
    }
}