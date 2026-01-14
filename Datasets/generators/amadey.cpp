#include <iostream>
#include <fstream>
static unsigned int seed = 192837465;
unsigned int nextRandom()
{
    seed = seed * 1103515245 + 12345;
    return seed;
}
char randomChar()
{
    // f-o, q-z, 1-9
    unsigned int r = nextRandom() % 3;
    if (r == 0) return 'f' + (nextRandom() % ('o'-'f'+1));
    if (r == 1) return 'q' + (nextRandom() % ('z'-'q'+1));
    return '1' + (nextRandom() % 9);
}
int main()
{
    std::ofstream out("/home/nafisa/Documents/SPL-1 Project Draft/Datasets/dga_raw/amadey.csv");
    if (!out.is_open())
    {
        std::cerr << "Failed to open amadey.csv\n";
        return 1;
    }
    out << "domain,label\n";
    const int NUM_SAMPLES = 5000; 
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        std::string domain;
        int len = 6 + (nextRandom() % 6); // 6-11 chars
        for (int j = 0; j < len; j++)
        {
            domain += randomChar();
        }
        domain += ".info";
        out << domain << ",1\n";
    }
    out.close();
    std::cout << "Amadey DGA dataset generated\n";
    std::cout << "Samples: " << NUM_SAMPLES << "\n";
    return 0;
}
