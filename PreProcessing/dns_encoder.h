#ifndef DNS_ENCODER_H
#define DNS_ENCODER_H
#include <vector>
#include <string>
#include "configure.h"
#include "encoder.h"
std::vector<int> encodeDns(const std::string &dns)
{
    std::vector<int> encoded;
    for(char c : dns)
    {
        if((int)encoded.size() >= maxLength) break;
        encoded.push_back(charToIndex(c));
    }
    while((int)encoded.size() < maxLength)
    {
        encoded.push_back(0);
    }
   return encoded; 
}
#endif
