#include <iostream>
#include <vector>
#include <map>
#include "csv_loader.h"
#include "dns_encoder.h"
#include "configure.h"
#include "label.h"
int main()
{
    // Data containers
    std::vector<std::vector<int>> X_train, X_test;
    std::vector<int> y_train, y_test;
    // Load merged dataset
    loadDataset("train.csv", X_train, y_train);
    loadDataset("test.csv", X_test, y_test);
    std::cout << "Train samples: " << X_train.size() << "\n";
    std::cout << "Test samples: " << X_test.size() << "\n";
    // Count labels
    std::map<int, long long> label_counts;
    std::map<int, std::vector<int>> examples; // store first 3 indices per class
    for (size_t i = 0; i < y_train.size(); i++)
    {
        int label = y_train[i];
        label_counts[label]++;
        if (examples[label].size() < 3) examples[label].push_back(i);
    }
    // Print counts
    std::cout << "\nTrain dataset stats:\n";
    std::cout << "Benign: " << label_counts[0] << "\n";
    std::cout << "DGA: " << label_counts[1] << "\n";
    std::cout << "Phishing: " << label_counts[2] << "\n";
    std::cout << "Tunneling: " << label_counts[3] << "\n";
    std::cout << "C2: " << label_counts[4] << "\n";
    // Print a few encoded examples per class
    std::cout << "\nSample encoded DNS per class:\n";
    for (const auto &pair : examples)
    {
    int label = pair.first;
    std::cout << "Label " << label << " examples:\n";
    for (int idx : pair.second)
    {
        for (int v : X_train[idx])
        {
            std::cout << v << " ";
        }
        std::cout << "\n";
    }
    }
    // First test example
    if (!X_test.empty())
    {
        std::cout << "\nFirst test encoded DNS:\n";
        for (int v : X_test[0]) std::cout << v << " ";
        std::cout << "\nLabel: " << y_test[0] << "\n";
    }
    return 0;
}
