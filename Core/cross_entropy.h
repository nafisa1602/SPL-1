#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H
namespace cross_entropy
{
  double categoricalCrossEntropy(const double *yTrue, const double *yPredic, int numClasses);
  void softmaxCrossEntroGrad(const double *logits, const double *yTrue, double *gradient, int numClasses);
}
#endif