#include <iostream>
#include <fstream>
#include <string>
static unsigned int seed = 135792468;
unsigned int nextRandom()
{
    seed = seed * 1103515245 + 12345;
    return seed;
}
char randomChar()
{
    const char* chars = "abcdefghijklmnopqrstuvwxyz0123456789";
    return chars[nextRandom() % 36];
}
int main()
{
    std::ofstream out("/home/nafisa/Documents/SPL-1 Project Draft/Datasets/dga_raw/allasenha.csv");
    if (!out.is_open())
    {
        std::cerr << "Failed to open allasenha.csv\n";
        return 1;
    }
    out << "domain,label\n";
    const int NUM_SAMPLES = 5000;
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        int len = 13 + (nextRandom() % (51 - 13 + 1)); // 13-51 chars
        std::string domain;
        for (int j = 0; j < len; j++)
        {
            domain += randomChar();
        }
        domain += ".brazilsouth.cloudapp.azure.com";
        out << domain << ",1\n";
    }
    out.close();
    std::cout << "AllaSenha DGA dataset generated\n";
    std::cout << "Samples: " << NUM_SAMPLES << "\n";
    return 0;
}
