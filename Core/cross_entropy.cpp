#include "cross_entropy.h"
#include "vector_math.h"
#include "advanced_math.h"

namespace cross_entropy
{
    double categoricalCrossEntropy(const double* yTrue, const double* yPred, int numClasses)
    {
        const double eps = 1e-9;
        double loss = 0.0;
        for(int i = 0; i < numClasses; i++)
        {
            double p = advanced_math::clamp(yPred[i], eps, 1.0 - eps);
            loss += -yTrue[i] * advanced_math::logarithm(p);
        }
        return loss;
    }

    double categoricalCrossEntropyFromLogits_OneHotY(
        int y, const double* logits, double* logProbsWorkspace, int numClasses
    )
    {
        vector_math::logSoftMax(logits, logProbsWorkspace, numClasses);
        return -logProbsWorkspace[y];
    }

    void softmaxCrossEntroGrad_OneHotY(
        const double* logits, int y, double* gradient, double* probsWorkspace, int numClasses
    )
    {
        vector_math::softMax(logits, probsWorkspace, numClasses);
        for(int i = 0; i < numClasses; i++) gradient[i] = probsWorkspace[i];
        gradient[y] -= 1.0;
    }
}
