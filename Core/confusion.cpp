#include <iostream>
#include <vector>
#include <cstdio>
#include <algorithm>

#include "configure.h"
#include "csv_loader.h"
#include "lstm.h"
#include "dense.h"

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

bool loadModel(const char* filename,
               double* fgW, double* igW, double* ogW, double* candW,
               double* fgB, double* igB, double* ogB, double* candB,
               int hiddenSize, int concatSize,
               double* classifierW, double* classifierB, int numClasses)
{
    auto readExact = [](FILE* file, void* dst, size_t elemSize, size_t count) -> bool
    {
        return std::fread(dst, elemSize, count, file) == count;
    };

    FILE* f = fopen(filename, "rb");
    if (!f) {
        std::cerr << "Failed to load model from " << filename << "\n";
        return false;
    }
    
    // Read and verify metadata
    int savedHidden, savedConcat, savedClasses;
    if (!readExact(f, &savedHidden, sizeof(int), 1) ||
        !readExact(f, &savedConcat, sizeof(int), 1) ||
        !readExact(f, &savedClasses, sizeof(int), 1))
    {
        std::cerr << "Failed to read model header from " << filename << "\n";
        fclose(f);
        return false;
    }
    
    if (savedHidden != hiddenSize || savedConcat != concatSize || savedClasses != numClasses) {
        std::cerr << "Model dimensions mismatch!\n";
        std::cerr << "Expected: hidden=" << hiddenSize << ", concat=" << concatSize << ", classes=" << numClasses << "\n";
        std::cerr << "Got: hidden=" << savedHidden << ", concat=" << savedConcat << ", classes=" << savedClasses << "\n";
        fclose(f);
        return false;
    }
    
    int totalWeights = hiddenSize * concatSize;
    
    // Read LSTM weights
    if (!readExact(f, fgW, sizeof(double), totalWeights) ||
        !readExact(f, igW, sizeof(double), totalWeights) ||
        !readExact(f, ogW, sizeof(double), totalWeights) ||
        !readExact(f, candW, sizeof(double), totalWeights))
    {
        std::cerr << "Failed to read LSTM weights from " << filename << "\n";
        fclose(f);
        return false;
    }
    
    // Read LSTM biases
    if (!readExact(f, fgB, sizeof(double), hiddenSize) ||
        !readExact(f, igB, sizeof(double), hiddenSize) ||
        !readExact(f, ogB, sizeof(double), hiddenSize) ||
        !readExact(f, candB, sizeof(double), hiddenSize))
    {
        std::cerr << "Failed to read LSTM biases from " << filename << "\n";
        fclose(f);
        return false;
    }
    
    // Read classifier weights and biases
    if (!readExact(f, classifierW, sizeof(double), hiddenSize * numClasses) ||
        !readExact(f, classifierB, sizeof(double), numClasses))
    {
        std::cerr << "Failed to read classifier parameters from " << filename << "\n";
        fclose(f);
        return false;
    }
    
    fclose(f);
    std::cout << "Model loaded from " << filename << "\n";
    return true;
}

int main(int argc, char* argv[])
{
    const char* modelPath = "best_model.bin";
    const char* testDataPath = "test.csv";
    
    // Allow command line arguments
    if (argc > 1) modelPath = argv[1];
    if (argc > 2) testDataPath = argv[2];
    
    const int T = config::kMaxLength;
    const int vocabSize = config::kVocabSize;
    const int inputSize = vocabSize;
    const int hiddenSize = config::kHiddenSize;
    const int numClasses = config::kNumClasses;
    const int concatSize = inputSize + hiddenSize;

    // Allocate model buffers with RAII to avoid manual delete[] on early returns.
    std::vector<double> forgetGateWeight(hiddenSize * concatSize);
    std::vector<double> inputGateWeight(hiddenSize * concatSize);
    std::vector<double> outputGateWeight(hiddenSize * concatSize);
    std::vector<double> candidateWeight(hiddenSize * concatSize);

    std::vector<double> forgetGateBias(hiddenSize);
    std::vector<double> inputGateBias(hiddenSize);
    std::vector<double> outputGateBias(hiddenSize);
    std::vector<double> candidateBias(hiddenSize);

    dense::denseLayer classifier(hiddenSize, numClasses);

    // Load the model
    std::cout << "Loading model from: " << modelPath << "\n";
    if (!loadModel(modelPath,
                   forgetGateWeight.data(), inputGateWeight.data(), outputGateWeight.data(), candidateWeight.data(),
                   forgetGateBias.data(), inputGateBias.data(), outputGateBias.data(), candidateBias.data(),
                   hiddenSize, concatSize,
                   classifier.weight, classifier.bias, numClasses))
    {
        std::cerr << "Failed to load model. Exiting.\n";
        return 1;
    }

    // Load test data
    std::vector<std::vector<int>> X_test;
    std::vector<int> y_test;
    
    std::cout << "Loading test data from: " << testDataPath << "\n";
    loadDataset(testDataPath, X_test, y_test);
    std::cout << "Test samples: " << X_test.size() << "\n\n";
    if (X_test.empty() || y_test.empty())
    {
        std::cerr << "No test data loaded. Exiting.\n";
        return 1;
    }

    // Allocate state
    lstmState state[T];
    for (int tt = 0; tt < T; tt++)
        initLstmState(state[tt], hiddenSize, concatSize);

    std::vector<double> x(inputSize);
    std::vector<double> logits(numClasses);
    const std::vector<double> hZero(hiddenSize, 0.0);
    const std::vector<double> cZero(hiddenSize, 0.0);

    // Confusion matrix
    int confusionMatrix[numClasses][numClasses] = {0};
    int correctTest = 0;

    std::cout << "========== Running Inference ==========\n";
    
    for (size_t sampleIdx = 0; sampleIdx < X_test.size(); sampleIdx++)
    {
        int lastIdx = lastNonZeroIndex(X_test[sampleIdx]);
        int Teff = lastIdx + 1;

        for (int tt = 0; tt < Teff; tt++)
        {
            std::fill(x.begin(), x.end(), 0.0);
            int idx = X_test[sampleIdx][tt];
            if (idx > 0 && idx < vocabSize)
                x[idx] = 1.0;

            const double* hPrev = (tt == 0) ? hZero.data() : state[tt - 1].hidden;
            const double* cPrev = (tt == 0) ? cZero.data() : state[tt - 1].cell;

            lstmForward(
                x.data(),
                hPrev,
                cPrev,
                forgetGateWeight.data(), forgetGateBias.data(),
                inputGateWeight.data(), inputGateBias.data(),
                outputGateWeight.data(), outputGateBias.data(),
                candidateWeight.data(), candidateBias.data(),
                inputSize,
                hiddenSize,
                state[tt]
            );
        }

        classifier.forward(state[Teff - 1].hidden, logits.data());

        int pred = argmax(logits.data(), numClasses);
        int actual = y_test[sampleIdx];
        
        if (pred == actual)
            correctTest++;
            
        if (actual >= 0 && actual < numClasses && pred >= 0 && pred < numClasses)
            confusionMatrix[actual][pred]++;
    }

    double testAcc = (double)correctTest / X_test.size();

    std::cout << "\n========== Results ==========\n";
    std::cout << "Test Accuracy: " << testAcc * 100.0 << "% (" << correctTest << "/" << X_test.size() << ")\n\n";

    // Print confusion matrix
    std::cout << "Confusion Matrix (rows=actual, cols=predicted):\n";
    std::cout << "     ";
    for (int i = 0; i < numClasses; i++)
        std::cout << "  [" << i << "]";
    std::cout << "\n";
    
    for (int i = 0; i < numClasses; i++)
    {
        std::cout << "[" << i << "] ";
        for (int j = 0; j < numClasses; j++) {
            std::cout << "  " << confusionMatrix[i][j] << "  ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nPer-class accuracy:\n";
    for (int i = 0; i < numClasses; i++)
    {
        int total = 0;
        for (int j = 0; j < numClasses; j++)
            total += confusionMatrix[i][j];
        
        if (total > 0) {
            double classAcc = (double)confusionMatrix[i][i] / total;
            std::cout << "Class " << i << ": " << classAcc * 100.0 << "% (" << confusionMatrix[i][i] << "/" << total << ")\n";
        }
    }

    // Cleanup
    for (int tt = 0; tt < T; tt++)
        freeLstmState(state[tt]);

    return 0;
}