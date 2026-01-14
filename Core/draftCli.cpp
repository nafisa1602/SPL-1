#include <iostream>
#include <vector>
#include "csv_loader.h"
#include "lstm.h"
#include "configure.h"
using namespace std;
double randUniform(double minValue, double maxValue)
{
    static unsigned int seed = 123456;
    seed = seed * 1103515245 + 12345;
    double val = ((seed / 65536) % 32768) / 32768.0;  
    return minValue + val * (maxValue - minValue);
}
void initWeights(double* w, int rows, int cols, double minVal, double maxVal)
{
    for(int i=0; i<rows*cols; i++)
        w[i] = randUniform(minVal, maxVal);
}
void initBias(double* b, int size, double minVal, double maxVal)
{
    for(int i=0; i<size; i++)
        b[i] = randUniform(minVal, maxVal);
}
int main()
{
    vector<vector<int>> X_train, X_test;
    vector<int> y_train, y_test;
    loadDataset("train.csv", X_train, y_train);
    loadDataset("test.csv", X_test, y_test);
    cout << "Train samples: " << X_train.size() << "\n";
    cout << "Test samples: " << X_test.size() << "\n";
    int hiddenSize = 8;              
    int concatSize = maxLength + hiddenSize;
    lstmState state;
    initLstmState(state, hiddenSize, concatSize);
    double hPrev[hiddenSize] = {0};
    double cPrev[hiddenSize] = {0};
    double fWeight[concatSize*hiddenSize], iWeight[concatSize*hiddenSize];
    double oWeight[concatSize*hiddenSize], cWeight[concatSize*hiddenSize];
    double fBias[hiddenSize], iBias[hiddenSize], oBias[hiddenSize], cBias[hiddenSize];
    // Larger weight ranges to avoid "flat" sigmoid
    initWeights(fWeight, hiddenSize, concatSize, -2.0, 2.0);
    initWeights(iWeight, hiddenSize, concatSize, -2.0, 2.0);
    initWeights(oWeight, hiddenSize, concatSize, -2.0, 2.0);
    initWeights(cWeight, hiddenSize, concatSize, -2.0, 2.0);
    initBias(fBias, hiddenSize, -1.0, 1.0);
    initBias(iBias, hiddenSize, -1.0, 1.0);
    initBias(oBias, hiddenSize, -1.0, 1.0);
    initBias(cBias, hiddenSize, -1.0, 1.0);
    cout << "\n=== Running LSTM forward on TRAIN data ===\n";
    for(size_t sample=0; sample < X_train.size(); sample++)
    {
        const vector<int>& dns = X_train[sample];
        for(size_t t=0; t<dns.size(); t++)
        {
            // Slightly higher scaling for more variation
            double x[1] = { static_cast<double>(dns[t]) / 5.0 };
            lstmForward(x, hPrev, cPrev, fWeight, fBias, iWeight, iBias, oWeight, oBias, cWeight, cBias,1, hiddenSize, state);
            for(int i=0;i<hiddenSize;i++)
            {
                hPrev[i] = state.hidden[i];
                cPrev[i] = state.cell[i];
            }
        }
        cout << "Sample " << sample << " last hidden: ";
        for(int i=0;i<hiddenSize;i++) cout << state.hidden[i] << " ";
        cout << "\n";
        // Reset previous state for next sample
        for(int i=0;i<hiddenSize;i++){ hPrev[i]=0; cPrev[i]=0; }
    }

    cout << "\n=== Running LSTM forward on TEST data ===\n";
    for(size_t sample=0; sample<X_test.size(); sample++)
    {
        const vector<int>& dns = X_test[sample];
        for(size_t t=0; t<dns.size(); t++)
        {
            double x[1] = { static_cast<double>(dns[t]) / 5.0 };
            lstmForward(x, hPrev, cPrev, fWeight, fBias, iWeight, iBias, oWeight, oBias, cWeight, cBias,1, hiddenSize, state);
            for(int i=0;i<hiddenSize;i++)
            {
                hPrev[i] = state.hidden[i];
                cPrev[i] = state.cell[i];
            }
        }
        cout << "Sample " << sample << " last hidden: ";
        for(int i=0;i<hiddenSize;i++) cout << state.hidden[i] << " ";
        cout << "\n";

        for(int i=0;i<hiddenSize;i++){ hPrev[i]=0; cPrev[i]=0; }
    }
    freeLstmState(state);
    return 0;
}
