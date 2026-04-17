#include "cross_entropy.h"
#include "vector_math.h"
#include "advanced_math.h"

namespace cross_entropy
{
double categoricalCrossEntropy(const double* yTrue, const double* yPred, int numClasses)
{
    const double eps = 1e-9;
    double loss = 0.0;

    for (int classIndex = 0; classIndex < numClasses; classIndex++) {
        double prob = advanced_math::clamp(yPred[classIndex], eps, 1.0 - eps);
        loss += -yTrue[classIndex] * advanced_math::logarithm(prob);
    }

    return loss;
}

double categoricalCrossEntropyFromLogits_OneHotY(
    int classIndex, const double* logits, double* logProbsWorkspace, int numClasses
)
{
    vector_math::logSoftMax(logits, logProbsWorkspace, numClasses);
    return -logProbsWorkspace[classIndex];
}

void softmaxCrossEntroGrad_OneHotY(
    const double* logits, int classIndex, double* gradient, double* probsWorkspace, int numClasses
)
{
    vector_math::softMax(logits, probsWorkspace, numClasses);
    for (int i = 0; i < numClasses; i++) {
        gradient[i] = probsWorkspace[i];
    }
    gradient[classIndex] -= 1.0;
}
}
