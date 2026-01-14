#include <iostream>
#include <fstream>
static unsigned int seed = 123456789;
unsigned int nextRandom()
{
    seed = seed * 1103515245 + 12345;
    return seed;
}
char randomLower()
{
    return 'a' + (nextRandom() % 26);
}
const char* randomSuffix()
{
    unsigned int r = nextRandom() % 3;

    if (r == 0) return "tk";
    if (r == 1) return "com";
    return "pages.dev";
}
int main()
{
    std::ofstream out("/home/nafisa/Documents/SPL-1 Project Draft/Datasets/dga_raw/abcbot.csv");
    if (!out.is_open())
    {
        std::cerr << "Failed to open abcbot.csv\n";
        return 1;
    }
    // CSV header
    out << "domain,label\n";
    const int NUM_SAMPLES = 10000;
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        char domain[64];
        int pos = 0;
        // generate 9 lowercase letters
        for (int j = 0; j < 9; j++)
        {
            domain[pos++] = randomLower();
        }
        domain[pos++] = '.';
        const char* suffix = randomSuffix();
        for (int k = 0; suffix[k] != '\0'; k++)
        {
            domain[pos++] = suffix[k];
        }

        domain[pos] = '\0';
        out << domain << ",1\n";
    }
    out.close();
    std::cout << "Abcbot DGA dataset generated\n";
    std::cout << "Samples: " << NUM_SAMPLES << "\n";
    return 0;
}
