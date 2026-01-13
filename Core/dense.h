#ifndef DENSE_H
#define DENSE_H
namespace dense
{
    struct denseLayer
    {
        int inputSize;
        int outputSize;
        double *weight;
        double *bias;
        double *inputCache;
        denseLayer(int input, int output);
        ~denseLayer();
        void forward(const double *input, double *output);
        void backward(const double *denseOutput, double *denseInput, double learningRate);
    };
}
#endif