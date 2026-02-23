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

static int argmax(const double* v, int n)
{
    int bestIdx = 0;
    double bestVal = v[0];
    for(int i = 1; i < n; i++)
    {
        if(v[i] > bestVal)
        {
            bestVal = v[i];
            bestIdx = i;
        }
    }
    return bestIdx;
}

static int lastNonZeroIndex(const std::vector<int>& seq)
{
    for(int i = (int)seq.size() - 1; i >= 0; i--)
        if(seq[i] != 0) return i;
    return 0;
}

int main()
{
    const int T = maxLength;
    const int vocabSize = 40;
    const int inputSize = vocabSize;
    const int hiddenSize = 32;
    const int numClasses = 5;

    const double learningRate = 0.0003;
    const int epochs = 10;
    const int truncK = 5;
    const double gradClip = 0.5;

    rng::seed(123456u);

    std::vector<std::vector<int>> X_train, X_test;
    std::vector<int> y_train, y_test;

    loadDataset("train.csv", X_train, y_train);
    loadDataset("test.csv", X_test, y_test);

    std::cout << "Train samples: " << X_train.size() << "\n";
    std::cout << "Test samples: " << X_test.size() << "\n";

    long long trainClassCount[numClasses] = {0,0,0,0,0};
    for(int y : y_train)
        if(y >= 0 && y < numClasses) trainClassCount[y]++;

    std::cout << "Train class counts:\n";
    for(int i = 0; i < numClasses; i++)
        std::cout << "Class " << i << ": " << trainClassCount[i] << "\n";

    
    long long testClassCount[numClasses] = {0,0,0,0,0};
    for(int y : y_test)
        if(y >= 0 && y < numClasses) testClassCount[y]++;

    std::cout << "Test class counts:\n";
    for(int i = 0; i < numClasses; i++)
        std::cout << "Class " << i << ": " << testClassCount[i] << "\n";

    std::vector<int> byClass[numClasses];
    for(int i = 0; i < (int)y_train.size(); i++)
    {
        int y = y_train[i];
        if(y >= 0 && y < numClasses) byClass[y].push_back(i);
    }

    
    double classWeight[numClasses];
    double total = (double)X_train.size();
    for(int c = 0; c < numClasses; c++)
    {
        double w = total / (numClasses * (double)trainClassCount[c]);
        if(w > 10.0) w = 10.0;
        if(w < 1.0) w = 1.0;
        classWeight[c] = 1.0 + 0.15 * (w - 1.0);
    }

    std::cout << "Tempered class weights (FYI):\n";
    for(int c = 0; c < numClasses; c++)
        std::cout << "w[" << c << "] = " << classWeight[c] << "\n";

    const int concatSize = inputSize + hiddenSize;

    double* forgetGateWeight = new double[hiddenSize * concatSize];
    double* inputGateWeight  = new double[hiddenSize * concatSize];
    double* outputGateWeight = new double[hiddenSize * concatSize];
    double* candidateWeight  = new double[hiddenSize * concatSize];

    double* forgetGateBias = new double[hiddenSize];
    double* inputGateBias  = new double[hiddenSize];
    double* outputGateBias = new double[hiddenSize];
    double* candidateBias  = new double[hiddenSize];

    for(int i = 0; i < hiddenSize * concatSize; i++)
    {
        forgetGateWeight[i] = rng::uniform(-0.05, 0.05);
        inputGateWeight[i]  = rng::uniform(-0.05, 0.05);
        outputGateWeight[i] = rng::uniform(-0.05, 0.05);
        candidateWeight[i]  = rng::uniform(-0.05, 0.05);
    }

    for(int i = 0; i < hiddenSize; i++)
    {
        forgetGateBias[i] = 1.0;
        inputGateBias[i] = 0.0;
        outputGateBias[i] = 0.0;
        candidateBias[i] = 0.0;
    }

    dense::denseLayer classifier(hiddenSize, numClasses);

    lstmState state[T];
    for(int tt = 0; tt < T; tt++)
        initLstmState(state[tt], hiddenSize, concatSize);

    std::vector<double> x(inputSize, 0.0);
    std::vector<double> logits(numClasses, 0.0);
    std::vector<double> probs(numClasses, 0.0);
    std::vector<double> logProbs(numClasses, 0.0);
    std::vector<double> dLogits(numClasses, 0.0);
    std::vector<double> dHiddenT(hiddenSize, 0.0);

    std::vector<double> hZero(hiddenSize, 0.0);
    std::vector<double> cZero(hiddenSize, 0.0);

    int stepsPerEpoch = (int)X_train.size();

    for(int epoch = 0; epoch < epochs; epoch++)
    {
        double totalLoss = 0.0;
        int correctTrain = 0;
        long long predCount[numClasses] = {0,0,0,0,0};
        int usedSteps = 0;

        for(int step = 0; step < stepsPerEpoch; step++)
        {
            // Balanced sampling by class
            int cls = (int)(rng::uniform01() * numClasses);
            if(cls < 0) cls = 0;
            if(cls >= numClasses) cls = numClasses - 1;
            if(byClass[cls].empty()) continue;

            int pick = (int)(rng::uniform01() * byClass[cls].size());
            if(pick < 0) pick = 0;
            if(pick >= (int)byClass[cls].size()) pick = (int)byClass[cls].size() - 1;

            int n = byClass[cls][pick];

            for(int i = 0; i < hiddenSize; i++)
            {
                hZero[i] = 0.0;
                cZero[i] = 0.0;
            }

            int lastIdx = lastNonZeroIndex(X_train[n]);
            int Teff = lastIdx + 1;

            for(int tt = 0; tt < Teff; tt++)
            {
                for(int i = 0; i < inputSize; i++) x[i] = 0.0;

                int idx = X_train[n][tt];
                if(idx > 0 && idx < vocabSize) x[idx] = 1.0;

                const double* hPrev = (tt == 0) ? hZero.data() : state[tt - 1].hidden;
                const double* cPrev = (tt == 0) ? cZero.data() : state[tt - 1].cell;

                lstmForward(
                    x.data(),
                    hPrev,
                    cPrev,
                    forgetGateWeight, forgetGateBias,
                    inputGateWeight, inputGateBias,
                    outputGateWeight, outputGateBias,
                    candidateWeight, candidateBias,
                    inputSize,
                    hiddenSize,
                    state[tt]
                );
            }

            classifier.forward(state[Teff - 1].hidden, logits.data());

            int y = y_train[n];
            if(y < 0 || y >= numClasses) continue;

            double sampleLoss = cross_entropy::categoricalCrossEntropyFromLogits_OneHotY(
                y, logits.data(), logProbs.data(), numClasses
            );

            usedSteps++;
            totalLoss += sampleLoss;

            cross_entropy::softmaxCrossEntroGrad_OneHotY(
                logits.data(), y, dLogits.data(), probs.data(), numClasses
            );

            // FIX: Do NOT scale by classWeight here because you already do balanced sampling.
            // (Remove the old dLogits *= classWeight[y] block.)

            classifier.backward(dLogits.data(), dHiddenT.data(), learningRate);

            int kEff = truncK;
            if(kEff > Teff) kEff = Teff;

            lstmBackwardTruncated(
                state,
                Teff,
                kEff,
                cZero.data(),
                dHiddenT.data(),
                forgetGateWeight, forgetGateBias,
                inputGateWeight, inputGateBias,
                outputGateWeight, outputGateBias,
                candidateWeight, candidateBias,
                inputSize,
                hiddenSize,
                learningRate,
                gradClip
            );

            int pred = argmax(logits.data(), numClasses);
            predCount[pred]++;

            if(pred == y) correctTrain++;
        }

        double denom = (usedSteps > 0) ? (double)usedSteps : 1.0;
        double trainAcc = (double)correctTrain / denom;
        double avgLoss = totalLoss / denom;

        int correctTest = 0;
        long long testTotalByClass2[numClasses]   = {0,0,0,0,0};
        long long testCorrectByClass[numClasses]  = {0,0,0,0,0};
        long long testPredByClass[numClasses]     = {0,0,0,0,0};

        for(size_t n = 0; n < X_test.size(); n++)
        {
            for(int i = 0; i < hiddenSize; i++)
            {
                hZero[i] = 0.0;
                cZero[i] = 0.0;
            }

            int lastIdx = lastNonZeroIndex(X_test[n]);
            int Teff = lastIdx + 1;

            for(int tt = 0; tt < Teff; tt++)
            {
                for(int i = 0; i < inputSize; i++) x[i] = 0.0;

                int idx = X_test[n][tt];
                if(idx > 0 && idx < vocabSize) x[idx] = 1.0;

                const double* hPrev = (tt == 0) ? hZero.data() : state[tt - 1].hidden;
                const double* cPrev = (tt == 0) ? cZero.data() : state[tt - 1].cell;

                lstmForward(
                    x.data(),
                    hPrev,
                    cPrev,
                    forgetGateWeight, forgetGateBias,
                    inputGateWeight, inputGateBias,
                    outputGateWeight, outputGateBias,
                    candidateWeight, candidateBias,
                    inputSize,
                    hiddenSize,
                    state[tt]
                );
            }

            classifier.forward(state[Teff - 1].hidden, logits.data());

            int pred = argmax(logits.data(), numClasses);
            int y = y_test[n];

            if(pred >= 0 && pred < numClasses) testPredByClass[pred]++;

            if(y >= 0 && y < numClasses)
            {
                testTotalByClass2[y]++;
                if(pred == y) testCorrectByClass[y]++;
            }

            if(pred == y) correctTest++;
        }

        double testAcc = (double)correctTest / (double)X_test.size();

        double macroRecall = 0.0;
        int macroCount = 0;
        for(int c = 0; c < numClasses; c++)
        {
            if(testTotalByClass2[c] > 0)
            {
                macroRecall += (double)testCorrectByClass[c] / (double)testTotalByClass2[c];
                macroCount++;
            }
        }
        if(macroCount > 0) macroRecall /= (double)macroCount;

        std::cout << "Epoch " << epoch
                  << " | Loss: " << avgLoss
                  << " | TrainAcc: " << trainAcc
                  << " | TestAcc: " << testAcc
                  << " | MacroRecall: " << macroRecall
                  << " | Pred(train): "
                  << predCount[0] << " "
                  << predCount[1] << " "
                  << predCount[2] << " "
                  << predCount[3] << " "
                  << predCount[4]
                  << "\n";

        std::cout << "Test recall per class: ";
        for(int c = 0; c < numClasses; c++)
        {
            double r = 0.0;
            if(testTotalByClass2[c] > 0)
                r = (double)testCorrectByClass[c] / (double)testTotalByClass2[c];
            std::cout << r << (c == numClasses - 1 ? "" : " ");
        }
        std::cout << "\n";

        // NEW: precision per class
        std::cout << "Test precision per class: ";
        for(int c = 0; c < numClasses; c++)
        {
            double p = 0.0;
            if(testPredByClass[c] > 0)
                p = (double)testCorrectByClass[c] / (double)testPredByClass[c];
            std::cout << p << (c == numClasses - 1 ? "" : " ");
        }
        std::cout << "\n";

        // NEW: prediction distribution on test
        std::cout << "Pred(test): "
                  << testPredByClass[0] << " "
                  << testPredByClass[1] << " "
                  << testPredByClass[2] << " "
                  << testPredByClass[3] << " "
                  << testPredByClass[4] << "\n";
    }

    for(int tt = 0; tt < T; tt++)
        freeLstmState(state[tt]);

    delete[] forgetGateWeight;
    delete[] inputGateWeight;
    delete[] outputGateWeight;
    delete[] candidateWeight;

    delete[] forgetGateBias;
    delete[] inputGateBias;
    delete[] outputGateBias;
    delete[] candidateBias;

    return 0;
}
