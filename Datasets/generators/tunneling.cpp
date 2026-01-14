#include <iostream>
#include <fstream>
static unsigned int seed = 987654321;
unsigned int nextRandom()
{
    seed = seed * 1103515245 + 12345;
    return seed;
}
char randomBase64Like()
{
    const char chars[] =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789";

    return chars[nextRandom() % (sizeof(chars) - 1)];
}
const char* baseDomain()
{
    unsigned int r = nextRandom() % 3;

    if (r == 0) return "tunnel.com";
    if (r == 1) return "dnsdata.net";
    return "exfil.org";
}
int main()
{
    std::ofstream out("/home/nafisa/Documents/SPL-1 Project Draft/Datasets/tunneling_raw/tunneling.csv");
    if (!out.is_open())
    {
        std::cerr << "Failed to open tunneling.csv\n";
        return 1;
    }
    // CSV header
    out << "domain,label\n";
    const int NUM_SAMPLES = 1000; // required minimum
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        char domain[128];
        int pos = 0;
        // long, high-entropy subdomain (typical tunneling)
        int subLen = 30 + (nextRandom() % 20);
        for (int j = 0; j < subLen; j++)
        {
            domain[pos++] = randomBase64Like();
        }
        domain[pos++] = '.';
        const char* base = baseDomain();
        for (int k = 0; base[k] != '\0'; k++)
        {
            domain[pos++] = base[k];
        }
        domain[pos] = '\0';
        out << domain << ",tunneling\n";
    }
    out.close();
    std::cout << "Tunneling DNS dataset generated\n";
    std::cout << "Samples: " << NUM_SAMPLES << "\n";
    return 0;
}
