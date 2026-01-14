#ifndef DNS_CLEANER_H
#define DNS_CLEANER_H
#include<string>
std::string cleanDns(const std::string &input)
{
    std::string out;
    for (char c : input)
    {
        if (c >= 'A' && c <= 'Z') 
        {
            c += ('a' - 'A');
        }
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '_')
        {
            out.push_back(c);
        }
    }
    if (!out.empty() && out.back() == '.') out.pop_back();
    return out;
}
#endif

