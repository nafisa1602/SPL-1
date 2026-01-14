#ifndef LABEL_H
#define LABEL_H
#include <string>
#include <iostream>
int labelToId(const std::string &label)
{
    if(label == "benign") return 0;
    if(label == "dga") return 1;
    if(label == "phishing") return 2;
    if(label == "tunneling") return 3;
    if(label == "c2") return 4;
    std::cerr << "Warning: unknown label '" << label << "' encountered\n";
    return -1;
}

#endif
