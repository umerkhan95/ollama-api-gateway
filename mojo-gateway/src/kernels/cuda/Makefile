# EdgeLLM CUDA Kernel Build System
# Builds shared libraries for NVIDIA GPUs

NVCC = nvcc
NVCC_FLAGS_COMMON = -O3 -Xcompiler -fPIC -Xcompiler -Wall

# CUDA architecture flags
# Jetson Nano: sm_53
# Jetson Xavier/Orin: sm_72, sm_87
# RTX 20 series: sm_75
# RTX 30 series: sm_86
# RTX 40 series: sm_89
CUDA_ARCH ?= -gencode arch=compute_53,code=sm_53 \
             -gencode arch=compute_72,code=sm_72 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_86,code=sm_86

# Platform detection
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    SHARED_EXT = dylib
    SHARED_FLAGS = -Xlinker -dynamiclib
else
    SHARED_EXT = so
    SHARED_FLAGS = -shared
endif

NVCC_FLAGS = $(NVCC_FLAGS_COMMON) $(CUDA_ARCH)

# Output directories
LIB_DIR = ../../../lib
BIN_DIR = ../../../bin

# Targets
TARGET = $(LIB_DIR)/libtmac_kernel_cuda.$(SHARED_EXT)
STATIC_TARGET = $(LIB_DIR)/libtmac_kernel_cuda.a

SOURCES = tmac_kernel.cu
OBJECTS = tmac_kernel.o

.PHONY: all clean test install dirs check info

# Default target
all: check dirs $(TARGET)

# Check for NVCC
check:
	@which nvcc > /dev/null 2>&1 || (echo "Error: nvcc not found. Install CUDA Toolkit." && exit 1)

dirs:
	@mkdir -p $(LIB_DIR) $(BIN_DIR)

$(TARGET): $(OBJECTS)
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built CUDA kernel: $(TARGET)"

$(STATIC_TARGET): $(OBJECTS)
	ar rcs $@ $^
	@echo "Built static library: $(STATIC_TARGET)"

%.o: %.cu tmac_kernel_cuda.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Test binary
test: dirs $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/test_cuda_kernel test_cuda_kernel.cu $(OBJECTS) -lcudart
	@echo "Running CUDA tests..."
	$(BIN_DIR)/test_cuda_kernel

# Clean
clean:
	rm -f $(OBJECTS) $(TARGET) $(STATIC_TARGET)
	rm -f $(BIN_DIR)/test_cuda_kernel

# Install to system (optional)
install: $(TARGET)
	@echo "Installing to /usr/local/lib..."
	cp $(TARGET) /usr/local/lib/
	cp tmac_kernel_cuda.h /usr/local/include/

# Jetson Nano specific build
jetson-nano:
	$(MAKE) CUDA_ARCH="-gencode arch=compute_53,code=sm_53"

# Jetson Orin specific build
jetson-orin:
	$(MAKE) CUDA_ARCH="-gencode arch=compute_87,code=sm_87"

# RTX build (for development)
rtx:
	$(MAKE) CUDA_ARCH="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89"

# Debug build
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: all

# Profiling build (for Nsight)
profile: NVCC_FLAGS += -lineinfo
profile: all

# Tensor Core build (INT8 support for sm_75+)
# Enables WMMA API for INT8 Tensor Core operations
tensorcore:
	$(MAKE) CUDA_ARCH="-gencode arch=compute_75,code=sm_75 \
	                   -gencode arch=compute_80,code=sm_80 \
	                   -gencode arch=compute_86,code=sm_86 \
	                   -gencode arch=compute_87,code=sm_87 \
	                   -gencode arch=compute_89,code=sm_89" \
	        NVCC_FLAGS_COMMON="$(NVCC_FLAGS_COMMON) --expt-relaxed-constexpr"

# T4 specific build (Kaggle/Colab)
t4:
	$(MAKE) CUDA_ARCH="-gencode arch=compute_75,code=sm_75" \
	        NVCC_FLAGS_COMMON="$(NVCC_FLAGS_COMMON) --expt-relaxed-constexpr"

# A100 specific build
a100:
	$(MAKE) CUDA_ARCH="-gencode arch=compute_80,code=sm_80" \
	        NVCC_FLAGS_COMMON="$(NVCC_FLAGS_COMMON) --expt-relaxed-constexpr"

# Print configuration
info:
	@echo "NVCC: $(shell which nvcc 2>/dev/null || echo 'not found')"
	@nvcc --version 2>/dev/null || echo "CUDA not available"
	@echo "Target: $(TARGET)"
	@echo "CUDA_ARCH: $(CUDA_ARCH)"
	@echo ""
	@echo "Build targets:"
	@echo "  make            - Default multi-arch build"
	@echo "  make tensorcore - INT8 Tensor Core optimized (sm_75+)"
	@echo "  make t4         - Tesla T4 optimized (Kaggle/Colab)"
	@echo "  make a100       - A100 optimized"
	@echo "  make jetson-nano - Jetson Nano (sm_53)"
	@echo "  make jetson-orin - Jetson Orin (sm_87)"
	@echo "  make rtx        - RTX 30/40 series"
