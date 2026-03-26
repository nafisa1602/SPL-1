#ifndef CSV_LOADER_H
#define CSV_LOADER_H
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
#include <cstdlib>
#include "dns_cleaner.h"
#include "dns_encoder.h"
#include "label.h"

// Helper function to find dataset path (portable, not hardcoded)
inline std::string getDatasetPath(const std::string &filename)
{
    // Try multiple possible locations:
    // 1. Relative to executable: ../Datasets/merged/
    // 2. Environment variable DATASET_PATH
    // 3. Current directory: ./Datasets/merged/
    
    namespace fs = std::filesystem;
    
    std::vector<std::string> searchPaths = {
        "../Datasets/merged/" + filename,
        "Datasets/merged/" + filename,
        "./Datasets/merged/" + filename
    };
    
    // Also check environment variable
    const char* envPath = std::getenv("DATASET_PATH");
    if(envPath) {
        searchPaths.insert(searchPaths.begin(), std::string(envPath) + "/" + filename);
    }
    
    for(const auto &path : searchPaths) {
        if(fs::exists(path)) {
            std::cerr << "[INFO] Using dataset: " << path << "\n";
            return path;
        }
    }
    
    // If no path found, return first option and let it fail gracefully
    std::cerr << "[ERROR] Dataset not found in any standard location. Tried:\n";
    for(const auto &path : searchPaths)
        std::cerr << "  - " << path << "\n";
    
    return searchPaths[0];
}

inline void loadDataset(const std::string &filename, std::vector<std::vector<int>> &X, std::vector<int> &y)
{
    std::string datasetPath = getDatasetPath(filename);
    std::ifstream file(datasetPath);
    if (!file.is_open())
    {
        std::cerr << "Cannot open file: " << datasetPath << "\n";
        return;
    }
    std::string line;
    // Skip header
    std::getline(file, line);
    int skippedEmpty = 0;
    int skippedLabel = 0;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string domain, label;

        std::getline(ss, domain, ',');
        std::getline(ss, label);
        if (domain.empty() || label.empty())
        {
            skippedEmpty++;
            continue;
        } // skip empty lines
        std::string clean = cleanDns(domain);
        std::vector<int> encoded = encodeDns(clean);
        int labelId = labelToId(label);
        if (labelId != -1)
        {
            X.push_back(encoded);
            y.push_back(labelId);
        }
        else
        {
            skippedLabel++;
        }
    }
    file.close();
    if(skippedEmpty > 0) std::cout << "Skipped " << skippedEmpty << " empty lines in  " << filename << "\n";
    if(skippedLabel > 0) std::cout << "Skipped " << skippedLabel << " unknown labels in " << filename << "\n";
}
#endif

