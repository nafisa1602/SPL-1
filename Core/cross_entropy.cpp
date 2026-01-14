#include "cross_entropy.h"
#include "vector_math.h"
#include "advanced_math.h"
namespace cross_entropy
{
    double categoricalCrossEntropy(const double* yTrue, const double* yPredic, int numClasses)
    {
        const double epsilon = 1e-9;   
        double loss = 0.0;
        for(int i = 0; i < numClasses; i++)
        {
            double probabilty = advanced_math::clamp(yPredic[i], epsilon, 1.0);
            loss += -yTrue[i] * advanced_math::logarithm(probabilty);
        }
        return loss;
    }

    void softmaxCrossEntroGrad(const double* logits, const double* yTrue, double* gradient, int numClasses)
    {
        double probabilities[numClasses];
        vector_math::softMax(logits, probabilities, numClasses);

        for(int i = 0; i < numClasses; i++)
        {
            gradient[i] = probabilities[i] - yTrue[i];
        }
    }
}
