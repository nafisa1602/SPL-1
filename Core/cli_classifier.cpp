#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include "configure.h"
#include "csv_loader.h"
#include "lstm.h"
#include "lstm_backward.h"
#include "dense.h"
#include "cross_entropy.h"
#include "rng.h"
#include "vector_math.h"
#include "advanced_math.h"
#include "basic_math.h"
#include <fstream>
#include <sstream>
static const char* CLASS_NAMES[5] = 
{
    "Benign", "DGA", "Phishing", "Tunneling", "C2"
};
static const char* CLASS_DESC[5] = 
{
    "Normal / safe traffic",
    "Domain Generation Algorithm (malware)",
    "Phishing domain",
    "DNS Tunneling (data exfiltration)",
    "Command & Control communication"
};
static constexpr int kClassCount = 5;
struct Model
{
    int hiddenSize;
    int concatSize;
    int numClasses;
 
    std::vector<double> fgW, igW, ogW, cW;
    std::vector<double> fgB, igB, ogB, cB;
    dense::denseLayer*  classifier;
 
    Model() : hiddenSize(0), concatSize(0), numClasses(0), classifier(nullptr) {}
    ~Model() { delete classifier; }
};
 
static bool loadModel(const char* path, Model& m)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        std::cerr << "[ERROR] Cannot open model file: " << path << "\n";
        return false;
    }
 
    auto readExact = [](FILE* f, void* dst, size_t sz, size_t n) -> bool {
        return fread(dst, sz, n, f) == n;
    };
 
    int hs, cs, nc;
    if (!readExact(f, &hs, sizeof(int), 1) ||
        !readExact(f, &cs, sizeof(int), 1) ||
        !readExact(f, &nc, sizeof(int), 1))
    {
        std::cerr << "[ERROR] Failed to read model header.\n";
        fclose(f); return false;
    }
 
    // Validate against compile-time config
    if (hs != config::kHiddenSize || nc != config::kNumClasses) {
        std::cerr << "[ERROR] Model dimensions mismatch.\n";
        std::cerr << "  Expected hidden=" << config::kHiddenSize
                  << " classes=" << config::kNumClasses << "\n";
        std::cerr << "  Got     hidden=" << hs << " classes=" << nc << "\n";
        fclose(f); return false;
    }
 
    m.hiddenSize = hs;
    m.concatSize = cs;
    m.numClasses = nc;
 
    int tw = hs * cs;
    m.fgW.resize(tw); m.igW.resize(tw);
    m.ogW.resize(tw); m.cW.resize(tw);
    m.fgB.resize(hs); m.igB.resize(hs);
    m.ogB.resize(hs); m.cB.resize(hs);
 
    if (!readExact(f, m.fgW.data(), sizeof(double), tw) ||
        !readExact(f, m.igW.data(), sizeof(double), tw) ||
        !readExact(f, m.ogW.data(), sizeof(double), tw) ||
        !readExact(f, m.cW.data(),  sizeof(double), tw) ||
        !readExact(f, m.fgB.data(), sizeof(double), hs) ||
        !readExact(f, m.igB.data(), sizeof(double), hs) ||
        !readExact(f, m.ogB.data(), sizeof(double), hs) ||
        !readExact(f, m.cB.data(),  sizeof(double), hs))
    {
        std::cerr << "[ERROR] Failed to read LSTM weights.\n";
        fclose(f); return false;
    }
 
    m.classifier = new dense::denseLayer(hs, nc);
    if (!readExact(f, m.classifier->weight, sizeof(double), hs * nc) ||
        !readExact(f, m.classifier->bias,   sizeof(double), nc))
    {
        std::cerr << "[ERROR] Failed to read classifier weights.\n";
        fclose(f); return false;
    }
 
    fclose(f);
    return true;
}

static int predict(const std::vector<int>& seq, const Model& m,
                   std::vector<double>& confidences)
{
    const int hs = m.hiddenSize;
    const int cs = m.concatSize;
    const int vs = config::kVocabSize;
 
    // Find effective length (last non-zero token)
    int Teff = 0;
    for (int i = (int)seq.size() - 1; i >= 0; i--)
        if (seq[i] != 0) { Teff = i + 1; break; }
    if (Teff == 0) Teff = 1;
 
    // Allocate LSTM states
    std::vector<lstmState> state(Teff);
    for (int t = 0; t < Teff; t++)
        initLstmState(state[t], hs, cs);
 
    std::vector<double> x(vs, 0.0);
    std::vector<double> hZero(hs, 0.0);
    std::vector<double> cZero(hs, 0.0);
 
    // Forward pass
    for (int t = 0; t < Teff; t++)
    {
        std::fill(x.begin(), x.end(), 0.0);
        int idx = seq[t];
        if (idx > 0 && idx < vs) x[idx] = 1.0;
 
        const double* hPrev = (t == 0) ? hZero.data() : state[t-1].hidden;
        const double* cPrev = (t == 0) ? cZero.data() : state[t-1].cell;
 
        lstmForward(x.data(), hPrev, cPrev,
            m.fgW.data(), m.fgB.data(),
            m.igW.data(), m.igB.data(),
            m.ogW.data(), m.ogB.data(),
            m.cW.data(),  m.cB.data(),
            vs, hs, state[t]);
    }
 
    // Dense classifier
    std::vector<double> logits(m.numClasses);
    m.classifier->forward(state[Teff-1].hidden, logits.data());
 
    // Softmax for confidence scores (stable and accurate)
    double maxL = logits[0];
    for (int i = 1; i < m.numClasses; i++)
        if (logits[i] > maxL) maxL = logits[i];
 
    double sumExp = 0.0;
    confidences.resize(m.numClasses);
    for (int i = 0; i < m.numClasses; i++) {
        double e = advanced_math::exponential(logits[i] - maxL);
        confidences[i] = e;
        sumExp += e;
    }
    for (int i = 0; i < m.numClasses; i++)
        confidences[i] /= sumExp;
 
    // Cleanup
    for (int t = 0; t < Teff; t++) freeLstmState(state[t]);
 
    // Argmax
    int best = 0;
    for (int i = 1; i < m.numClasses; i++)
        if (confidences[i] > confidences[best]) best = i;
    return best;
}

static void printResult(const std::string& domain, int cls,
                         const std::vector<double>& conf, bool verbose)
{
    std::cout << "\n";
    std::cout << "  Domain     : " << domain << "\n";
    std::cout << "  Prediction : " << CLASS_NAMES[cls] << "\n";
    std::cout << "  Description: " << CLASS_DESC[cls] << "\n";
    std::cout << "  Confidence : " << (int)(conf[cls] * 100.0 + 0.5) << "%\n";
 
    if (verbose)
    {
        std::cout << "\n  All class scores:\n";
        for (int i = 0; i < kClassCount; i++)
        {
            int pct = (int)(conf[i] * 100.0 + 0.5);
            std::cout << "    [" << i << "] " << CLASS_NAMES[i];
            // pad to alignment
            int nameLen = (int)strlen(CLASS_NAMES[i]);
            for (int s = nameLen; s < 10; s++) std::cout << ' ';
            std::cout << ": ";
            // ASCII bar
            int bars = pct / 5;
            std::cout << "[";
            for (int b = 0; b < 20; b++) std::cout << (b < bars ? "#" : " ");
            std::cout << "] " << pct << "%\n";
        }
    }
    std::cout << "\n";
}
 
static void printSeparator()
{
    std::cout << "─────────────────────────────────────────────\n";
}

static void runBatch(const char* csvPath, const Model& m, bool verbose)
{
    std::ifstream f(csvPath);
    if (!f.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << csvPath << "\n";
        return;
    }
 
    int total = 0, unknown = 0;
    std::string line;
    bool hasHeader = true;
 
    // Count results per class
    int perClass[kClassCount] = {0, 0, 0, 0, 0};
 
    std::cout << "\n========== Batch Analysis: " << csvPath << " ==========\n";
 
    while (std::getline(f, line))
    {
        if (line.empty()) continue;
 
        // Parse: domain[,label]
        std::string domain, labelStr;
        std::istringstream ss(line);
        std::getline(ss, domain, ',');
        std::getline(ss, labelStr);
 
        // Skip header row
        if (hasHeader && (domain == "domain" || domain == "Domain" || domain == "query")) {
            hasHeader = false;
            continue;
        }
        hasHeader = false;
 
        if (domain.empty()) continue;
 
        std::string cleaned = cleanDns(domain);
        if (cleaned.empty()) { unknown++; continue; }
 
        std::vector<int> seq = encodeDns(cleaned);
        std::vector<double> conf;
        int cls = predict(seq, m, conf);
 
        perClass[cls]++;
        total++;
 
        if (verbose) {
            printSeparator();
            printResult(domain, cls, conf, false);
        } else {
            // compact output
            std::cout << domain << "  →  " << CLASS_NAMES[cls]
                      << " (" << (int)(conf[cls]*100+0.5) << "%)\n";
        }
    }
 
    // Summary
    std::cout << "\n========== Summary ==========\n";
    std::cout << "Total processed : " << total << "\n";
    if (unknown > 0)
        std::cout << "Skipped (empty) : " << unknown << "\n";
    std::cout << "\nClass distribution:\n";
    for (int i = 0; i < kClassCount; i++)
        std::cout << "  " << CLASS_NAMES[i] << ": " << perClass[i] << "\n";
    std::cout << "\n";
}
 
static void runInteractive(const Model& m)
{
    std::cout << "\n";
    printSeparator();
    std::cout << "  DNS Threat Detector  |  Interactive Mode\n";
    std::cout << "  Type a domain name and press Enter.\n";
    std::cout << "  Commands: 'verbose' (toggle detail) | 'quit' (exit)\n";
    printSeparator();
 
    bool verbose = true;
    std::string input;
 
    while (true)
    {
        std::cout << "\n> ";
        if (!std::getline(std::cin, input)) break;
 
        // Trim whitespace
        size_t start = input.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = input.find_last_not_of(" \t\r\n");
        input = input.substr(start, end - start + 1);
 
        if (input.empty()) continue;
        if (input == "quit" || input == "exit" || input == "q") break;
        if (input == "verbose") {
            verbose = !verbose;
            std::cout << "  Verbose mode " << (verbose ? "ON" : "OFF") << "\n";
            continue;
        }
        if (input == "help") {
            std::cout << "  Commands:\n";
            std::cout << "    verbose  - toggle detailed confidence scores\n";
            std::cout << "    quit     - exit the program\n";
            std::cout << "  Just type any domain name to classify it.\n";
            continue;
        }
 
        std::string cleaned = cleanDns(input);
        if (cleaned.empty()) {
            std::cout << "  [!] Invalid domain — no recognizable characters.\n";
            continue;
        }
 
        std::vector<int> seq = encodeDns(cleaned);
        std::vector<double> conf;
        int cls = predict(seq, m, conf);
        printResult(input, cls, conf, verbose);
    }
 
    std::cout << "\nGoodbye!\n";
}

static void printHelp(const char* prog)
{
    std::cout << "\n";
    std::cout << "dns_detect — DNS Threat Detector\n";
    std::cout << "SPL-1 Project | IIT, University of Dhaka\n\n";
    std::cout << "USAGE:\n";
    std::cout << "  " << prog << "                          Interactive mode\n";
    std::cout << "  " << prog << " -q <domain>              Classify a single domain\n";
    std::cout << "  " << prog << " -f <file.csv>            Batch classify domains from CSV\n";
    std::cout << "  " << prog << " -m <model.bin>           Specify model file (default: best_model.bin)\n";
    std::cout << "  " << prog << " -v                       Verbose output (show all class scores)\n";
    std::cout << "  " << prog << " --help                   Show this help\n\n";
    std::cout << "EXAMPLES:\n";
    std::cout << "  " << prog << " -q google.com\n";
    std::cout << "  " << prog << " -q xn--abc.malware.xyz\n";
    std::cout << "  " << prog << " -f domains.csv\n";
    std::cout << "  " << prog << " -m best_model.bin -f domains.csv -v\n\n";
    std::cout << "CLASSES:\n";
    for (int i = 0; i < kClassCount; i++)
        std::cout << "  [" << i << "] " << CLASS_NAMES[i] << " — " << CLASS_DESC[i] << "\n";
    std::cout << "\n";
}

int main(int argc, char* argv[])
{
    const char* modelPath = "best_model.bin";
    const char* queryDomain = nullptr;
    const char* batchFile   = nullptr;
    bool verbose = false;
 
    // Parse arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printHelp(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-m") == 0 && i+1 < argc) {
            modelPath = argv[++i];
        }
        else if (strcmp(argv[i], "-q") == 0 && i+1 < argc) {
            queryDomain = argv[++i];
        }
        else if (strcmp(argv[i], "-f") == 0 && i+1 < argc) {
            batchFile = argv[++i];
        }
        else if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        }
        else {
            // Treat bare argument as a domain query
            queryDomain = argv[i];
        }
    }
 
    // Load model
    std::cout << "Loading model: " << modelPath << " ... ";
    Model m;
    if (!loadModel(modelPath, m)) return 1;
    std::cout << "OK  (hidden=" << m.hiddenSize
              << ", classes=" << m.numClasses << ")\n";
 
    // Dispatch
    if (queryDomain)
    {
        std::string cleaned = cleanDns(queryDomain);
        if (cleaned.empty()) {
            std::cerr << "[ERROR] Domain has no valid characters after cleaning.\n";
            return 1;
        }
        std::vector<int> seq = encodeDns(cleaned);
        std::vector<double> conf;
        int cls = predict(seq, m, conf);
        printSeparator();
        printResult(queryDomain, cls, conf, true); // always verbose for single query
        printSeparator();
    }
    else if (batchFile)
    {
        runBatch(batchFile, m, verbose);
    }
    else
    {
        runInteractive(m);
    }
 
    return 0;
}