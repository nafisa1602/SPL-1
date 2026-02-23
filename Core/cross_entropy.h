#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

namespace cross_entropy
{
    double categoricalCrossEntropy(const double* yTrue, const double* yPred, int numClasses);

    double categoricalCrossEntropyFromLogits_OneHotY(
        int y, const double* logits, double* logProbsWorkspace, int numClasses
    );

    void softmaxCrossEntroGrad_OneHotY(
        const double* logits, int y, double* gradient, double* probsWorkspace, int numClasses
    );
}

#endif
