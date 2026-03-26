CXX = g++
CXXFLAGS = -fdiagnostics-color=always -g -std=c++17 -O2
INCLUDES = -I../PreProcessing
SOURCES = dga_classifier.cpp advanced_math.cpp basic_math.cpp cross_entropy.cpp dense.cpp lstm.cpp lstm_backward.cpp matrix_math.cpp rng.cpp vector_math.cpp
TARGET = dga_classifier

.PHONY: all clean help

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET)
	@echo "[✓] Build complete: $(TARGET)"

rebuild: clean all

clean:
	rm -f $(TARGET)
	@echo "[✓] Cleaned"

help:
	@echo "Makefile targets:"
	@echo "  make           - Build the CLI tool"
	@echo "  make rebuild   - Clean and rebuild"
	@echo "  make clean     - Remove binary"
	@echo "  make help      - Show this message"
