#include <iostream>
#include <string>
#include <vector>
#include <cstring>
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

// ============================================================================
// SHARED UTILITIES
// ============================================================================

static int argmax(const double* v, int n) {
    int bestIdx = 0;
    double bestVal = v[0];
    for(int i = 1; i < n; i++)
        if(v[i] > bestVal) { bestVal = v[i]; bestIdx = i; }
    return bestIdx;
}

static int lastNonZeroIndex(const std::vector<int>& seq) {
    for(int i = (int)seq.size() - 1; i >= 0; i--)
        if(seq[i] != 0) return i;
    return 0;
}

static void vFill(double* v, int n, double val) {
    for(int i = 0; i < n; i++) v[i] = val;
}

static void vCopy(const double* src, double* dst, int n) {
    for(int i = 0; i < n; i++) dst[i] = src[i];
}

static int minInt(int a, int b) { return (a < b) ? a : b; }

// ============================================================================
// MODEL SAVE/LOAD
// ============================================================================

void saveModel(const char* filename,
               const double* fgW, const double* igW, const double* ogW, const double* candW,
               const double* fgB, const double* igB, const double* ogB, const double* candB,
               int hiddenSize, int concatSize,
               const double* classifierW, const double* classifierB, int numClasses) {
    std::ofstream f(filename, std::ios::binary);
    if(!f) { std::cerr << "Failed to save model to " << filename << "\n"; return; }

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
    std::cout << "[✓] Model saved to " << filename << "\n";
}

bool loadModel(const char* filename,
               double* fgW, double* igW, double* ogW, double* candW,
               double* fgB, double* igB, double* ogB, double* candB,
               int hiddenSize, int concatSize,
               double* classifierW, double* classifierB, int numClasses) {
    auto readExact = [](FILE* file, void* dst, size_t elemSize, size_t count) -> bool {
        return std::fread(dst, elemSize, count, file) == count;
    };

    FILE* f = fopen(filename, "rb");
    if(!f) {
        std::cerr << "[✗] Failed to load model from " << filename << "\n";
        return false;
    }

    int savedHidden, savedConcat, savedClasses;
    if(!readExact(f, &savedHidden, sizeof(int), 1) ||
       !readExact(f, &savedConcat, sizeof(int), 1) ||
       !readExact(f, &savedClasses, sizeof(int), 1)) {
        std::cerr << "[✗] Failed to read model header from " << filename << "\n";
        fclose(f);
        return false;
    }

    if(savedHidden != hiddenSize || savedConcat != concatSize || savedClasses != numClasses) {
        std::cerr << "[✗] Model dimensions mismatch!\n";
        std::cerr << "   Expected: hidden=" << hiddenSize << ", concat=" << concatSize << ", classes=" << numClasses << "\n";
        std::cerr << "   Got: hidden=" << savedHidden << ", concat=" << savedConcat << ", classes=" << savedClasses << "\n";
        fclose(f);
        return false;
    }

    int totalWeights = hiddenSize * concatSize;
    if(!readExact(f, fgW, sizeof(double), totalWeights) ||
       !readExact(f, igW, sizeof(double), totalWeights) ||
       !readExact(f, ogW, sizeof(double), totalWeights) ||
       !readExact(f, candW, sizeof(double), totalWeights)) {
        std::cerr << "[✗] Failed to read LSTM weights from " << filename << "\n";
        fclose(f);
        return false;
    }

    if(!readExact(f, fgB, sizeof(double), hiddenSize) ||
       !readExact(f, igB, sizeof(double), hiddenSize) ||
       !readExact(f, ogB, sizeof(double), hiddenSize) ||
       !readExact(f, candB, sizeof(double), hiddenSize)) {
        std::cerr << "[✗] Failed to read LSTM biases from " << filename << "\n";
        fclose(f);
        return false;
    }

    if(!readExact(f, classifierW, sizeof(double), hiddenSize * numClasses) ||
       !readExact(f, classifierB, sizeof(double), numClasses)) {
        std::cerr << "[✗] Failed to read classifier parameters from " << filename << "\n";
        fclose(f);
        return false;
    }

    fclose(f);
    std::cout << "[✓] Model loaded from " << filename << "\n";
    return true;
}

// ============================================================================
// TRAINING
// ============================================================================

void applyDropout(double* vec, double* mask, int size, double dropRate) {
    double scale = 1.0 / (1.0 - dropRate);
    for(int i = 0; i < size; i++) {
        if(rng::uniform01() < dropRate) { vec[i] = 0.0; mask[i] = 0.0; }
        else { vec[i] *= scale; mask[i] = scale; }
    }
}

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
    std::vector<double>& x, std::vector<double>& logits) {
    int correct = 0;
    for(int n = 0; n < (int)X.size(); n++) {
        int Teff = lastNonZeroIndex(X[n]) + 1;
        for(int tt = 0; tt < Teff; tt++) {
            vFill(state[tt].hidden, hiddenSize, 0.0);
            vFill(state[tt].cell, hiddenSize, 0.0);
            vFill(state[tt].forget, hiddenSize, 0.0);
            vFill(state[tt].inputGate, hiddenSize, 0.0);
            vFill(state[tt].outputGate, hiddenSize, 0.0);
            vFill(state[tt].candidate, hiddenSize, 0.0);
            vFill(state[tt].concat, concatSize, 0.0);
        }
        for(int tt = 0; tt < Teff; tt++) {
            vFill(x.data(), vocabSize, 0.0);
            int idx = X[n][tt];
            if(idx > 0 && idx < vocabSize) x[idx] = 1.0;
            const double* hPrev = (tt == 0) ? hZero : state[tt-1].hidden;
            const double* cPrev = (tt == 0) ? cZero : state[tt-1].cell;
            lstmForward(x.data(), hPrev, cPrev,
                fGateWeight, fGateBias,
                iGateWeight, iGateBias,
                oGateWeight, oGateBias,
                canGateWeight, canGateBias,
                vocabSize, hiddenSize, state[tt]);
        }
        classifier.forward(state[Teff-1].hidden, logits.data());
        if(argmax(logits.data(), numClasses) == y[n]) correct++;
    }
    return correct;
}

int train_command(int argc, char* argv[]) {
    std::string trainPath = "train.csv";
    std::string testPath = "test.csv";
    std::string modelPath = "best_model.bin";
    int epochs = 40;

    // Parse arguments
    for(int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if(arg == "--train" && i+1 < argc) trainPath = argv[++i];
        else if(arg == "--test" && i+1 < argc) testPath = argv[++i];
        else if(arg == "--model" && i+1 < argc) modelPath = argv[++i];
        else if(arg == "--epochs" && i+1 < argc) epochs = std::atoi(argv[++i]);
    }

    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║    DGA Classifier - Training Mode      ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    std::cout << "Training dataset: " << trainPath << "\n";
    std::cout << "Test dataset:     " << testPath << "\n";
    std::cout << "Model output:     " << modelPath << "\n";
    std::cout << "Epochs:           " << epochs << "\n\n";

    const int T = config::kMaxLength;
    const int vocabSize = config::kVocabSize;
    const int inputSize = vocabSize;
    const int hiddenSize = config::kHiddenSize;
    const int numClasses = config::kNumClasses;
    const double initialLR = 0.001;
    const int truncK = 15;
    const double gradClip = 1.0;
    const double dropout_rate = 0.3;
    const int patience = 8;

    rng::seed(123456u);

    std::vector<std::vector<int>> X_train, X_test;
    std::vector<int> y_train, y_test;
    loadDataset(trainPath.c_str(), X_train, y_train);
    loadDataset(testPath.c_str(), X_test, y_test);
    std::cout << "[✓] Train samples: " << X_train.size() << "\n";
    std::cout << "[✓] Test samples:  " << X_test.size() << "\n\n";

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
    for(int i = 0; i < hiddenSize * concatSize; i++) {
        forgetGateWeight[i] = rng::uniform(-lstmStddev, lstmStddev);
        inputGateWeight[i]  = rng::uniform(-lstmStddev, lstmStddev);
        outputGateWeight[i] = rng::uniform(-lstmStddev, lstmStddev);
        candidateWeight[i]  = rng::uniform(-lstmStddev, lstmStddev);
    }
    for(int i = 0; i < hiddenSize; i++) {
        forgetGateBias[i] = 1.0;
        inputGateBias[i]  = 0.0;
        outputGateBias[i] = 0.0;
        candidateBias[i]  = 0.0;
    }

    dense::denseLayer classifier(hiddenSize, numClasses);

    lstmState state[T];
    for(int tt = 0; tt < T; tt++)
        initLstmState(state[tt], hiddenSize, concatSize);

    std::vector<double> x(inputSize), logits(numClasses), probs(numClasses);
    std::vector<double> logProbs(numClasses), dLogits(numClasses), dHiddenT(hiddenSize);
    std::vector<double> hiddenDropped(hiddenSize), dropoutMask(hiddenSize);
    const std::vector<double> hZero(hiddenSize, 0.0), cZero(hiddenSize, 0.0);

    vCopy(forgetGateWeight, best_fgW, hiddenSize * concatSize);
    vCopy(inputGateWeight,  best_igW, hiddenSize * concatSize);
    vCopy(outputGateWeight, best_ogW, hiddenSize * concatSize);
    vCopy(candidateWeight,  best_cW,  hiddenSize * concatSize);
    vCopy(forgetGateBias,   best_fgB, hiddenSize);
    vCopy(inputGateBias,    best_igB, hiddenSize);
    vCopy(outputGateBias,   best_ogB, hiddenSize);
    vCopy(candidateBias,    best_cB,  hiddenSize);

    std::vector<int> byClass[numClasses];
    for(int i = 0; i < (int)y_train.size(); i++) {
        int y = y_train[i];
        if(y >= 0 && y < numClasses) byClass[y].push_back(i);
    }

    double bestTestAcc  = 0.0;
    double prevAvgLoss  = 1e9;
    int patienceCounter = 0;
    int bestEpoch       = 0;

    std::cout << "========== Training Started ==========\n\n";

    for(int epoch = 0; epoch < epochs; epoch++) {
        double currentLR = initialLR * basic_math::power(0.95, epoch);
        double totalLoss = 0.0;
        int correctTrain = 0, usedSteps = 0;

        for(int step = 0; step < (int)X_train.size(); step++) {
            int cls = (int)(rng::uniform01() * numClasses);
            if(byClass[cls].empty()) continue;
            int pick = (int)(rng::uniform01() * byClass[cls].size());
            int n    = byClass[cls][pick];

            int Teff = lastNonZeroIndex(X_train[n]) + 1;

            for(int tt = 0; tt < Teff; tt++) {
                vFill(state[tt].hidden, hiddenSize, 0.0);
                vFill(state[tt].cell, hiddenSize, 0.0);
                vFill(state[tt].forget, hiddenSize, 0.0);
                vFill(state[tt].inputGate, hiddenSize, 0.0);
                vFill(state[tt].outputGate, hiddenSize, 0.0);
                vFill(state[tt].candidate, hiddenSize, 0.0);
                vFill(state[tt].concat, concatSize, 0.0);
            }

            for(int tt = 0; tt < Teff; tt++) {
                vFill(x.data(), inputSize, 0.0);
                int idx = X_train[n][tt];
                if(idx > 0 && idx < vocabSize) x[idx] = 1.0;
                const double* hPrev = (tt == 0) ? hZero.data() : state[tt-1].hidden;
                const double* cPrev = (tt == 0) ? cZero.data() : state[tt-1].cell;
                lstmForward(x.data(), hPrev, cPrev,
                    forgetGateWeight, forgetGateBias,
                    inputGateWeight,  inputGateBias,
                    outputGateWeight, outputGateBias,
                    candidateWeight,  candidateBias,
                    inputSize, hiddenSize, state[tt]);
            }

            vCopy(state[Teff-1].hidden, hiddenDropped.data(), hiddenSize);
            applyDropout(hiddenDropped.data(), dropoutMask.data(), hiddenSize, dropout_rate);
            classifier.forward(hiddenDropped.data(), logits.data());

            int y = y_train[n];
            if(y < 0 || y >= numClasses) continue;

            double loss = cross_entropy::categoricalCrossEntropyFromLogits_OneHotY(
                y, logits.data(), logProbs.data(), numClasses);

            if(loss != loss || loss > 1e6) continue;

            totalLoss += loss;
            usedSteps++;

            cross_entropy::softmaxCrossEntroGrad_OneHotY(
                logits.data(), y, dLogits.data(), probs.data(), numClasses);

            classifier.backward(dLogits.data(), dHiddenT.data(), currentLR);

            for(int i = 0; i < hiddenSize; i++)
                dHiddenT[i] *= dropoutMask[i];

            lstmBackwardTruncated(
                state, Teff, minInt(truncK, Teff), cZero.data(), dHiddenT.data(),
                forgetGateWeight, forgetGateBias,
                inputGateWeight,  inputGateBias,
                outputGateWeight, outputGateBias,
                candidateWeight,  candidateBias,
                inputSize, hiddenSize, currentLR, gradClip);

            if(argmax(logits.data(), numClasses) == y) correctTrain++;
        }

        double trainAcc = (usedSteps > 0) ? (double)correctTrain / usedSteps : 0.0;
        double avgLoss  = (usedSteps > 0) ? totalLoss / usedSteps : 0.0;

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
        std::cout << "Epoch " << epoch << " | LR: " << currentLR << " | Loss: " << avgLoss 
                  << " | Train: " << trainAcc << " | Test: " << testAcc;

        if(testAcc > bestTestAcc) {
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
            saveModel(modelPath.c_str(),
                forgetGateWeight, inputGateWeight, outputGateWeight, candidateWeight,
                forgetGateBias,   inputGateBias,   outputGateBias,   candidateBias,
                hiddenSize, concatSize, classifier.weight, classifier.bias, numClasses);
            std::cout << " ← BEST";
        } else {
            patienceCounter++;
            if(patienceCounter >= patience) {
                std::cout << "\n\n[!] Early stopping (no improvement for " << patience << " epochs)\n";
                break;
            }
        }
        std::cout << "\n";
    }

    std::cout << "\n========== Training Complete ==========\n";
    std::cout << "[✓] Best accuracy: " << bestTestAcc << " (Epoch " << bestEpoch << ")\n\n";

    delete[] forgetGateWeight; delete[] inputGateWeight;
    delete[] outputGateWeight; delete[] candidateWeight;
    delete[] forgetGateBias;   delete[] inputGateBias;
    delete[] outputGateBias;   delete[] candidateBias;
    delete[] best_fgW; delete[] best_igW;
    delete[] best_ogW; delete[] best_cW;
    delete[] best_fgB; delete[] best_igB;
    delete[] best_ogB; delete[] best_cB;

    for(int tt = 0; tt < T; tt++) freeLstmState(state[tt]);

    return 0;
}

// ============================================================================
// EVALUATION
// ============================================================================

int evaluate_command(int argc, char* argv[]) {
    std::string modelPath = "best_model.bin";
    std::string testPath = "test.csv";

    // Parse arguments
    for(int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if(arg == "--model" && i+1 < argc) modelPath = argv[++i];
        else if(arg == "--test" && i+1 < argc) testPath = argv[++i];
    }

    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║   DGA Classifier - Evaluation Mode     ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    std::cout << "Model:  " << modelPath << "\n";
    std::cout << "Test:   " << testPath << "\n\n";

    const int T = config::kMaxLength;
    const int vocabSize = config::kVocabSize;
    const int inputSize = vocabSize;
    const int hiddenSize = config::kHiddenSize;
    const int numClasses = config::kNumClasses;
    const int concatSize = inputSize + hiddenSize;

    std::vector<double> forgetGateWeight(hiddenSize * concatSize);
    std::vector<double> inputGateWeight(hiddenSize * concatSize);
    std::vector<double> outputGateWeight(hiddenSize * concatSize);
    std::vector<double> candidateWeight(hiddenSize * concatSize);
    std::vector<double> forgetGateBias(hiddenSize);
    std::vector<double> inputGateBias(hiddenSize);
    std::vector<double> outputGateBias(hiddenSize);
    std::vector<double> candidateBias(hiddenSize);

    dense::denseLayer classifier(hiddenSize, numClasses);

    if(!loadModel(modelPath.c_str(),
                  forgetGateWeight.data(), inputGateWeight.data(), outputGateWeight.data(), candidateWeight.data(),
                  forgetGateBias.data(), inputGateBias.data(), outputGateBias.data(), candidateBias.data(),
                  hiddenSize, concatSize,
                  classifier.weight, classifier.bias, numClasses)) {
        return 1;
    }

    std::vector<std::vector<int>> X_test;
    std::vector<int> y_test;
    loadDataset(testPath.c_str(), X_test, y_test);
    std::cout << "[✓] Test samples: " << X_test.size() << "\n\n";

    if(X_test.empty() || y_test.empty()) {
        std::cerr << "[✗] No test data loaded.\n";
        return 1;
    }

    lstmState state[T];
    for(int tt = 0; tt < T; tt++)
        initLstmState(state[tt], hiddenSize, concatSize);

    std::vector<double> x(inputSize);
    std::vector<double> logits(numClasses);
    const std::vector<double> hZero(hiddenSize, 0.0);
    const std::vector<double> cZero(hiddenSize, 0.0);

    int confusionMatrix[numClasses][numClasses];
    memset(confusionMatrix, 0, sizeof(confusionMatrix));
    int correctTest = 0;

    std::cout << "========== Running Inference ==========\n\n";

    for(size_t n = 0; n < X_test.size(); n++) {
        if(n % 50000 == 0 && n > 0)
            std::cout << "[.] Processed " << n << "/" << X_test.size() << "\n";

        int lastIdx = lastNonZeroIndex(X_test[n]);
        int Teff = lastIdx + 1;

        for(int tt = 0; tt < Teff; tt++) {
            std::fill(x.begin(), x.end(), 0.0);
            int idx = X_test[n][tt];
            if(idx > 0 && idx < vocabSize) x[idx] = 1.0;

            const double* hPrev = (tt == 0) ? hZero.data() : state[tt - 1].hidden;
            const double* cPrev = (tt == 0) ? cZero.data() : state[tt - 1].cell;

            lstmForward(x.data(), hPrev, cPrev,
                forgetGateWeight.data(), forgetGateBias.data(),
                inputGateWeight.data(), inputGateBias.data(),
                outputGateWeight.data(), outputGateBias.data(),
                candidateWeight.data(), candidateBias.data(),
                inputSize, hiddenSize, state[tt]);
        }

        classifier.forward(state[Teff - 1].hidden, logits.data());

        int prediction = argmax(logits.data(), numClasses);
        int actual = y_test[n];

        confusionMatrix[actual][prediction]++;
        if(prediction == actual) correctTest++;
    }

    std::cout << "\n========== Results ==========\n\n";

    const char* classNames[] = {"Benign", "DGA", "C2"};
    std::cout << "Confusion Matrix:\n\n";
    std::cout << "       ";
    for(int j = 0; j < numClasses; j++)
        std::cout << "  " << classNames[j];
    std::cout << "\n";

    for(int i = 0; i < numClasses; i++) {
        std::cout << classNames[i] << ": ";
        for(int j = 0; j < numClasses; j++)
            std::cout << "  " << confusionMatrix[i][j];
        std::cout << "\n";
    }

    double accuracy = (double)correctTest / X_test.size() * 100.0;
    double errorRate = 100.0 - accuracy;

    std::cout << "\n[✓] Accuracy:   " << accuracy << "%\n";
    std::cout << "[✓] Error Rate: " << errorRate << "%\n";
    std::cout << "[✓] Correct:    " << correctTest << "/" << X_test.size() << "\n\n";

    for(int tt = 0; tt < T; tt++) freeLstmState(state[tt]);

    return 0;
}

// ============================================================================
// HELP & MAIN
// ============================================================================

void print_help() {
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       DGA Classifier - LSTM-based DNS Query Classifier      ║\n";
    std::cout << "║                                                            ║\n";
    std::cout << "║  Classifies DNS domains as: Benign, DGA, or C2 Traffic     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "USAGE:\n";
    std::cout << "  dga_classifier <command> [options]\n\n";

    std::cout << "COMMANDS:\n\n";

    std::cout << "  train\n";
    std::cout << "    Train the LSTM model on DNS query data\n";
    std::cout << "    Options:\n";
    std::cout << "      --train <file>    Training CSV file (default: train.csv)\n";
    std::cout << "      --test <file>     Test CSV file (default: test.csv)\n";
    std::cout << "      --model <file>    Output model path (default: best_model.bin)\n";
    std::cout << "      --epochs <n>      Number of epochs (default: 40)\n";
    std::cout << "    Example:\n";
    std::cout << "      dga_classifier train --train train.csv --test test.csv --epochs 50\n\n";

    std::cout << "  evaluate\n";
    std::cout << "    Evaluate model on test dataset with confusion matrix\n";
    std::cout << "    Options:\n";
    std::cout << "      --model <file>    Model binary file (default: best_model.bin)\n";
    std::cout << "      --test <file>     Test CSV file (default: test.csv)\n";
    std::cout << "    Example:\n";
    std::cout << "      dga_classifier evaluate --model best_model.bin --test test.csv\n\n";

    std::cout << "  help\n";
    std::cout << "    Show this help message\n\n";

    std::cout << "DATASET FORMAT:\n";
    std::cout << "  CSV with columns: domain,class\n";
    std::cout << "  Classes: 0=Benign, 1=DGA, 2=C2\n";
    std::cout << "  Example:\n";
    std::cout << "    domain,class\n";
    std::cout << "    example.com,0\n";
    std::cout << "    abcbot.net,1\n";
    std::cout << "    c2server.org,2\n\n";
}

int main(int argc, char* argv[]) {
    if(argc < 2) {
        print_help();
        return 0;
    }

    std::string command = argv[1];

    if(command == "train") {
        return train_command(argc, argv);
    }
    else if(command == "evaluate") {
        return evaluate_command(argc, argv);
    }
    else if(command == "help" || command == "--help" || command == "-h") {
        print_help();
        return 0;
    }
    else {
        std::cerr << "[✗] Unknown command: " << command << "\n";
        std::cerr << "Use 'dga_classifier help' for usage information.\n\n";
        return 1;
    }
}
