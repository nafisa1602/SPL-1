#include <iostream>
#include <vector>
#include "configure.h"
#include "csv_loader.h"
#include "lstm.h"
#include "lstm_backward.h"
#include "dense.h"
#include "cross_entropy.h"
#include "rng.h"
#include "vector_math.h"
#include "advanced_math.h"
#include "basic_math.h"

static int argmax(const double* v, int n)
{
    int bestIdx = 0;
    double bestVal = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > bestVal) {
            bestVal = v[i];
            bestIdx = i;
        }
    }
    return bestIdx;
}

static int lastNonZeroIndex(const std::vector<int>& seq)
{
    for (int i = (int)seq.size() - 1; i >= 0; i--) {
        if (seq[i] != 0) return i;
    }
    return 0;
}

static void vFill(double* v, int n, double val)
{
    for (int i = 0; i < n; i++) {
        v[i] = val;
    }
}

static void vCopy(const double* src, double* dst, int n)
{
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

static int minInt(int a, int b) { return (a < b) ? a : b; }

void applyDropout(double* vec, double* mask, int size, double dropRate)
{
    double scale = 1.0 / (1.0 - dropRate);
    for (int i = 0; i < size; i++) {
        if (rng::uniform01() < dropRate) {
            vec[i] = 0.0;
            mask[i] = 0.0;
        } else {
            vec[i] *= scale;
            mask[i] = scale;
        }
    }
}

void saveModel(const char* filename,
               const double* fgW, const double* igW, const double* ogW, const double* candW,
               const double* fgB, const double* igB, const double* ogB, const double* candB,
               int hiddenSize, int concatSize,
               const double* classifierW, const double* classifierB, int numClasses)
{
    std::ofstream f(filename, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to save model to " << filename << "\n";
        return;
    }

    f.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(int));
    f.write(reinterpret_cast<const char*>(&concatSize),  sizeof(int));
    f.write(reinterpret_cast<const char*>(&numClasses),  sizeof(int));

    int tw = hiddenSize * concatSize;
    f.write(reinterpret_cast<const char*>(fgW),         sizeof(double) * tw);
    f.write(reinterpret_cast<const char*>(igW),         sizeof(double) * tw);
    f.write(reinterpret_cast<const char*>(ogW),         sizeof(double) * tw);
    f.write(reinterpret_cast<const char*>(candW),       sizeof(double) * tw);
    f.write(reinterpret_cast<const char*>(fgB),         sizeof(double) * hiddenSize);
    f.write(reinterpret_cast<const char*>(igB),         sizeof(double) * hiddenSize);
    f.write(reinterpret_cast<const char*>(ogB),         sizeof(double) * hiddenSize);
    f.write(reinterpret_cast<const char*>(candB),       sizeof(double) * hiddenSize);
    f.write(reinterpret_cast<const char*>(classifierW), sizeof(double) * hiddenSize * numClasses);
    f.write(reinterpret_cast<const char*>(classifierB), sizeof(double) * numClasses);
    std::cout << "Model saved to " << filename << "\n";
}

// Helper function to evaluate model on a dataset (shared by training and testing)
static int evaluateDataset(
    const std::vector<std::vector<int>>& X,
    const std::vector<int>& y,
    lstmState* state,
    const double* hZero, const double* cZero,
    int hiddenSize, int concatSize, int vocabSize, int numClasses,
    const double* fGateWeight, const double* fGateBias,
    const double* iGateWeight, const double* iGateBias,
    const double* oGateWeight, const double* oGateBias,
    const double* canGateWeight, const double* canGateBias,
    dense::denseLayer& classifier,
    std::vector<double>& x, std::vector<double>& logits)
{
    int correct = 0;
    for (int sampleIdx = 0; sampleIdx < (int)X.size(); sampleIdx++)
    {
        int Teff = lastNonZeroIndex(X[sampleIdx]) + 1;

        for (int tt = 0; tt < Teff; tt++)
        {
            vFill(state[tt].hidden,    hiddenSize, 0.0);
            vFill(state[tt].cell,      hiddenSize, 0.0);
            vFill(state[tt].forget,    hiddenSize, 0.0);
            vFill(state[tt].inputGate, hiddenSize, 0.0);
            vFill(state[tt].outputGate,hiddenSize, 0.0);
            vFill(state[tt].candidate, hiddenSize, 0.0);
            vFill(state[tt].concat,    concatSize, 0.0);
        }

        for (int tt = 0; tt < Teff; tt++)
        {
            vFill(x.data(), vocabSize, 0.0);
            int idx = X[sampleIdx][tt];
            if (idx > 0 && idx < vocabSize) x[idx] = 1.0;
            
            const double* hPrev = (tt == 0) ? hZero : state[tt - 1].hidden;
            const double* cPrev = (tt == 0) ? cZero : state[tt - 1].cell;
            lstmForward(x.data(), hPrev, cPrev,
                fGateWeight, fGateBias,
                iGateWeight, iGateBias,
                oGateWeight, oGateBias,
                canGateWeight, canGateBias,
                vocabSize, hiddenSize, state[tt]);
        }

        classifier.forward(state[Teff - 1].hidden, logits.data());
        if (argmax(logits.data(), numClasses) == y[sampleIdx]) {
            correct++;
        }
    }
    return correct;
}

int main()
{
    const int    T            = config::kMaxLength;
    const int    vocabSize    = config::kVocabSize;
    const int    inputSize    = vocabSize;
    const int    hiddenSize   = config::kHiddenSize;
    const int    numClasses   = config::kNumClasses;
    const double initialLR    = 0.001;
    const int    epochs       = 40;
    const int    truncK       = 15;
    const double gradClip     = 1.0;
    const double dropoutRate  = 0.3;
    const int    patience     = 8;

    rng::seed(123456u);

    std::vector<std::vector<int>> X_train, X_test;
    std::vector<int> y_train, y_test;
    loadDataset("train.csv", X_train, y_train);
    loadDataset("test.csv",  X_test,  y_test);
    std::cout << "Train samples: " << X_train.size() << "\n";
    std::cout << "Test samples: "  << X_test.size()  << "\n";

    std::vector<int> byClass[numClasses];
    for (int i = 0; i < (int)y_train.size(); i++) {
        int y = y_train[i];
        if (y >= 0 && y < numClasses) byClass[y].push_back(i);
    }

    const int concatSize = inputSize + hiddenSize;

    double* forgetGateWeight = new double[hiddenSize * concatSize];
    double* inputGateWeight  = new double[hiddenSize * concatSize];
    double* outputGateWeight = new double[hiddenSize * concatSize];
    double* candidateWeight  = new double[hiddenSize * concatSize];
    double* forgetGateBias   = new double[hiddenSize];
    double* inputGateBias    = new double[hiddenSize];
    double* outputGateBias   = new double[hiddenSize];
    double* candidateBias    = new double[hiddenSize];

    double* best_fgW = new double[hiddenSize * concatSize];
    double* best_igW = new double[hiddenSize * concatSize];
    double* best_ogW = new double[hiddenSize * concatSize];
    double* best_cW  = new double[hiddenSize * concatSize];
    double* best_fgB = new double[hiddenSize];
    double* best_igB = new double[hiddenSize];
    double* best_ogB = new double[hiddenSize];
    double* best_cB  = new double[hiddenSize];

    double lstmStddev = advanced_math::squareRoot(2.0 / (double)(inputSize + hiddenSize));
    for (int i = 0; i < hiddenSize * concatSize; i++) {
        forgetGateWeight[i] = rng::uniform(-lstmStddev, lstmStddev);
        inputGateWeight[i]  = rng::uniform(-lstmStddev, lstmStddev);
        outputGateWeight[i] = rng::uniform(-lstmStddev, lstmStddev);
        candidateWeight[i]  = rng::uniform(-lstmStddev, lstmStddev);
    }
    for (int i = 0; i < hiddenSize; i++) {
        forgetGateBias[i] = 1.0;  // reasonable default: sigmoid(1)=0.73
        inputGateBias[i]  = 0.0;
        outputGateBias[i] = 0.0;
        candidateBias[i]  = 0.0;
    }

    dense::denseLayer classifier(hiddenSize, numClasses);

    lstmState state[T];
    for (int tt = 0; tt < T; tt++)
        initLstmState(state[tt], hiddenSize, concatSize);

    std::vector<double> x(inputSize), logits(numClasses), probs(numClasses);
    std::vector<double> logProbs(numClasses), dLogits(numClasses), dHiddenT(hiddenSize);
    std::vector<double> hiddenDropped(hiddenSize), dropoutMask(hiddenSize);
    const std::vector<double> hZero(hiddenSize, 0.0), cZero(hiddenSize, 0.0);

    // Initialise best_* to starting weights so spike detector always has valid weights to restore
    vCopy(forgetGateWeight, best_fgW, hiddenSize * concatSize);
    vCopy(inputGateWeight,  best_igW, hiddenSize * concatSize);
    vCopy(outputGateWeight, best_ogW, hiddenSize * concatSize);
    vCopy(candidateWeight,  best_cW,  hiddenSize * concatSize);
    vCopy(forgetGateBias,   best_fgB, hiddenSize);
    vCopy(inputGateBias,    best_igB, hiddenSize);
    vCopy(outputGateBias,   best_ogB, hiddenSize);
    vCopy(candidateBias,    best_cB,  hiddenSize);

    double bestTestAcc  = 0.0;
    double prevAvgLoss  = 1e9;
    int patienceCounter = 0;
    int bestEpoch       = 0;

    std::cout << "\n========== Training Started ==========\n\n";

    for (int epoch = 0; epoch < epochs; epoch++) {
        double currentLR = initialLR * basic_math::power(0.95, epoch);
        double totalLoss = 0.0;
        int correctTrain = 0, usedSteps = 0;
        double maxWeightSeen = 0.0;  // track max weight magnitude this epoch
        double maxHiddenSeen = 0.0;  // track max hidden value this epoch

        for (int step = 0; step < (int)X_train.size(); step++) {
            int cls = (int)(rng::uniform01() * numClasses);
            if (byClass[cls].empty()) continue;
            int pick = (int)(rng::uniform01() * byClass[cls].size());
            int n    = byClass[cls][pick];

            int Teff = lastNonZeroIndex(X_train[n]) + 1;

            // Reset state array before each sample; stale gate values from
            // the previous sample corrupt both forward activations and BPTT
            for (int tt = 0; tt < Teff; tt++) {
                vFill(state[tt].hidden,    hiddenSize, 0.0);
                vFill(state[tt].cell,      hiddenSize, 0.0);
                vFill(state[tt].forget,    hiddenSize, 0.0);
                vFill(state[tt].inputGate, hiddenSize, 0.0);
                vFill(state[tt].outputGate,hiddenSize, 0.0);
                vFill(state[tt].candidate, hiddenSize, 0.0);
                vFill(state[tt].concat,    concatSize, 0.0);
            }

            for (int tt = 0; tt < Teff; tt++) {
                vFill(x.data(), inputSize, 0.0);
                int idx = X_train[n][tt];
                if (idx > 0 && idx < vocabSize) x[idx] = 1.0;

                const double* hPrev = (tt == 0) ? hZero.data() : state[tt - 1].hidden;
                const double* cPrev = (tt == 0) ? cZero.data() : state[tt - 1].cell;
                lstmForward(x.data(), hPrev, cPrev,
                    forgetGateWeight, forgetGateBias,
                    inputGateWeight,  inputGateBias,
                    outputGateWeight, outputGateBias,
                    candidateWeight,  candidateBias,
                    inputSize, hiddenSize, state[tt]);
            }

            vCopy(state[Teff-1].hidden, hiddenDropped.data(), hiddenSize);

            // Track max hidden value to detect saturation
            for (int i = 0; i < hiddenSize; i++) {
                double ah = state[Teff-1].hidden[i];
                if (ah < 0) ah = -ah;
                if (ah > maxHiddenSeen) maxHiddenSeen = ah;
            }
            applyDropout(hiddenDropped.data(), dropoutMask.data(), hiddenSize, dropoutRate);
            classifier.forward(hiddenDropped.data(), logits.data());

            int y = y_train[n];
            if (y < 0 || y >= numClasses) continue;

            double loss = cross_entropy::categoricalCrossEntropyFromLogits_OneHotY(
                y, logits.data(), logProbs.data(), numClasses);

            // Guard against NaN/exploded loss
            if (loss != loss || loss > 1e6) {
                std::cout << "[WARNING] Loss explosion detected at epoch " << epoch 
                          << ", step " << step << ", class " << y 
                          << ", loss=" << loss << ", logits=[";
                for (int i = 0; i < numClasses; i++)
                    std::cout << logits[i] << (i + 1 < numClasses ? "," : "");
                std::cout << "]\n";
                continue;  // Skip this sample
            }

            // Detect sudden per-sample loss spike
            if (epoch >= 10 && loss > 20.0) {
                std::cout << "[SPIKE] epoch=" << epoch << " step=" << step
                          << " y=" << y << " loss=" << loss
                          << " logits=";
                for (int i = 0; i < numClasses; i++) std::cout << logits[i] << " ";
                std::cout << "\n";
            }

            totalLoss += loss;
            usedSteps++;

            cross_entropy::softmaxCrossEntroGrad_OneHotY(
                logits.data(), y, dLogits.data(), probs.data(), numClasses);

            // For per-sample SGD, use full LR (gradient clipping handles explosion)
            classifier.backward(dLogits.data(), dHiddenT.data(), currentLR);

            for (int i = 0; i < hiddenSize; i++)
                dHiddenT[i] *= dropoutMask[i];

            lstmBackwardTruncated(
                state, Teff, minInt(truncK, Teff), cZero.data(), dHiddenT.data(),
                forgetGateWeight, forgetGateBias,
                inputGateWeight,  inputGateBias,
                outputGateWeight, outputGateBias,
                candidateWeight,  candidateBias,
                inputSize, hiddenSize, currentLR, gradClip);

            // Track max weight magnitude
            for (int i = 0; i < hiddenSize * concatSize; i++) {
                double aw = forgetGateWeight[i];
                if (aw < 0) aw = -aw;
                if (aw > maxWeightSeen) maxWeightSeen = aw;
                aw = inputGateWeight[i];
                if (aw < 0) aw = -aw;
                if (aw > maxWeightSeen) maxWeightSeen = aw;
            }

            if (argmax(logits.data(), numClasses) == y) correctTrain++;
        }

        double trainAcc = (usedSteps > 0) ? (double)correctTrain / usedSteps : 0.0;
        double avgLoss  = (usedSteps > 0) ? totalLoss / usedSteps : 0.0;

        std::cout << "[DEBUG] Epoch " << epoch
                  << " | usedSteps=" << usedSteps
                  << " | totalLoss=" << totalLoss
                  << " | avgLoss=" << avgLoss
                  << " | maxWeight=" << maxWeightSeen
                  << " | maxHidden=" << maxHiddenSeen << "\n";

        // Spike detection: restore best weights if loss jumps >50%
        if (avgLoss > prevAvgLoss * 1.5) {
            std::cout << "Loss spike detected (" << avgLoss << "), restoring best weights\n";
            vCopy(best_fgW, forgetGateWeight, hiddenSize * concatSize);
            vCopy(best_igW, inputGateWeight,  hiddenSize * concatSize);
            vCopy(best_ogW, outputGateWeight, hiddenSize * concatSize);
            vCopy(best_cW,  candidateWeight,  hiddenSize * concatSize);
            vCopy(best_fgB, forgetGateBias,   hiddenSize);
            vCopy(best_igB, inputGateBias,    hiddenSize);
            vCopy(best_ogB, outputGateBias,   hiddenSize);
            vCopy(best_cB,  candidateBias,    hiddenSize);
            patienceCounter++;
        } else {
            prevAvgLoss = avgLoss;
        }

        // Evaluate on test set (no dropout)
        int correctTest = evaluateDataset(
            X_test, y_test, state,
            hZero.data(), cZero.data(),
            hiddenSize, concatSize, vocabSize, numClasses,
            forgetGateWeight, forgetGateBias,
            inputGateWeight, inputGateBias,
            outputGateWeight, outputGateBias,
            candidateWeight, candidateBias,
            classifier, x, logits);

        double testAcc = (double)correctTest / X_test.size();
        std::cout << "Epoch " << epoch
                  << " | LR: "       << currentLR
                  << " | Loss: "     << avgLoss
                  << " | TrainAcc: " << trainAcc
                  << " | TestAcc: "  << testAcc;

        if (testAcc > bestTestAcc) {
            bestTestAcc     = testAcc;
            bestEpoch       = epoch;
            patienceCounter = 0;
            vCopy(forgetGateWeight, best_fgW, hiddenSize * concatSize);
            vCopy(inputGateWeight,  best_igW, hiddenSize * concatSize);
            vCopy(outputGateWeight, best_ogW, hiddenSize * concatSize);
            vCopy(candidateWeight,  best_cW,  hiddenSize * concatSize);
            vCopy(forgetGateBias,   best_fgB, hiddenSize);
            vCopy(inputGateBias,    best_igB, hiddenSize);
            vCopy(outputGateBias,   best_ogB, hiddenSize);
            vCopy(candidateBias,    best_cB,  hiddenSize);
            saveModel("best_model.bin",
                forgetGateWeight, inputGateWeight, outputGateWeight, candidateWeight,
                forgetGateBias,   inputGateBias,   outputGateBias,   candidateBias,
                hiddenSize, concatSize, classifier.weight, classifier.bias, numClasses);
            std::cout << " NEW BEST!";
        } else {
            patienceCounter++;
            if (patienceCounter >= patience) {
                std::cout << "\n\nEarly stopping triggered (no improvement for "
                          << patience << " epochs)\n";
                break;
            }
        }
        std::cout << "\n";
    }

    std::cout << "\n========== Training Completed ==========\n";
    std::cout << "Best Test Accuracy: " << bestTestAcc << " (Epoch " << bestEpoch << ")\n";
    std::cout << "Model saved to: best_model.bin\n\n";

    for (int tt = 0; tt < T; tt++) freeLstmState(state[tt]);

    delete[] forgetGateWeight; delete[] inputGateWeight;
    delete[] outputGateWeight; delete[] candidateWeight;
    delete[] forgetGateBias;   delete[] inputGateBias;
    delete[] outputGateBias;   delete[] candidateBias;
    delete[] best_fgW; delete[] best_igW;
    delete[] best_ogW; delete[] best_cW;
    delete[] best_fgB; delete[] best_igB;
    delete[] best_ogB; delete[] best_cB;

    return 0;
}