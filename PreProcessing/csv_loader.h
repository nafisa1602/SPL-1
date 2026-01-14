#ifndef CSV_LOADER_H
#define CSV_LOADER_H
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "dns_cleaner.h"
#include "dns_encoder.h"
#include "label.h"
void loadDataset(const std::string &filename, std::vector<std::vector<int>> &X, std::vector<int> &y)
{
    std::ifstream file("/home/nafisa/Documents/SPL-1 Project Draft/Datasets/merged/" + filename); // path to merged dataset
    if (!file.is_open())
    {
        std::cerr << "Cannot open file: " << filename << "\n";
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
    if(skippedLabel > 0) std::cout << "Skipped " << skippedLabel << " unlnown labels in " << filename << "\n";
}
#endif

