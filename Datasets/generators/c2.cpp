#include <iostream>
#include <fstream>
static unsigned int seed = 192837465;
unsigned int nextRandom()
{
    seed = seed * 1103515245 + 12345;
    return seed;
}
char randomLower()
{
    return 'a' + (nextRandom() % 26);
}
char randomDigit()
{
    return '0' + (nextRandom() % 10);
}
const char* c2Base()
{
    unsigned int r = nextRandom() % 3;

    if (r == 0) return "update-server.com";
    if (r == 1) return "cmd-control.net";
    return "node-manager.org";
}
int main()
{
    std::ofstream out("/home/nafisa/Documents/SPL-1 Project Draft/Datasets/c2_raw/c2.csv");
    if (!out.is_open())
    {
        std::cerr << "Failed to open c2.csv\n";
        return 1;
    }
    // CSV header
    out << "domain,label\n";
    const int NUM_SAMPLES = 1000;
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        char domain[128];
        int pos = 0;
        // medium-length subdomain (typical C2 beacon)
        int subLen = 10 + (nextRandom() % 6);
        for (int j = 0; j < subLen; j++)
        {
            if (nextRandom() % 2 == 0) domain[pos++] = randomLower();
            else domain[pos++] = randomDigit();
        }
        domain[pos++] = '.';
        const char* base = c2Base();
        for (int k = 0; base[k] != '\0'; k++)
        {
            domain[pos++] = base[k];
        }
        domain[pos] = '\0';
        out << domain << ",c2\n";
    }
    out.close();
    std::cout << "C2 DNS dataset generated\n";
    std::cout << "Samples: " << NUM_SAMPLES << "\n";
    return 0;
}
