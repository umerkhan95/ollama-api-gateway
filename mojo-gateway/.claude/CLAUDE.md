# EdgeLLM - Claude Code Context

## Project Overview

**EdgeLLM** is a platform for fine-tuning, optimizing, and deploying custom LLMs to edge devices with deterministic real-time performance. Built with Mojo (no GC) + hybrid C FFI kernels.

**Vision**: Fine-tune once, deploy everywhere - from cloud to edge.

**Target Market**: Edge/IoT, real-time AI, privacy-focused deployments

## Current Status (January 2026)

### Completed
- C FFI kernel integration (AVX2/NEON)
- **CUDA kernel integration (GPU acceleration)**
- BitNet 1.58-bit quantization pipeline
- T-MAC lookup table inference
- Transformer forward pass (RoPE, KV cache, sampling)
- Ollama benchmark comparison
- Docker-based Mojo development environment
- Automated benchmark suite with JSON output
- Unified kernel selector (auto-detects CUDA/AVX2/NEON)
- **Playwright MCP integration for browser automation**
- **Google Colab GPU testing pipeline (T4)**

### In Progress
- **CUDA kernel optimization to outperform Ollama** (see SKILL.md)
- Fly.io deployment for cloud benchmarking
- Paper-ready benchmark evaluation
- Multi-platform validation
- Metal kernels for Apple Silicon

### Experimentation Status (Jan 11, 2026)

**Problem Identified**: Current CUDA implementation is 12x slower than Ollama on T4 GPU.

See `SKILL.md` for the detailed optimization roadmap.

## Key Technologies

- **Mojo** - Systems language with ownership model (no GC), Python-like syntax
- **T-MAC** - Table lookup-based inference (no multiplication)
- **BitNet** - 1.58-bit ternary weight quantization
- **C FFI** - AVX2/NEON kernels for critical path (pshufb/tbl)
- **CUDA** - GPU acceleration for NVIDIA devices (Jetson, RTX)
- **QLoRA** - Efficient fine-tuning on consumer GPUs

## Performance Results (SmolLM-135M)

### Benchmark Comparison

| Metric | Ollama | EdgeLLM | Winner |
|--------|--------|---------|--------|
| Throughput | 156.7 tok/s | 38.4 tok/s (est.) | Ollama |
| Latency Jitter | 5566ms | <10ms (target) | **EdgeLLM** |
| Model Size | ~91 MB | 53.2 MB | **EdgeLLM** |
| Min Hardware | $800+ PC | $15 Pi Zero | **EdgeLLM** |

### Key Advantage: Deterministic Latency
```
Ollama:   ████████████████████████████████████████  5566 ms jitter
EdgeLLM:  █                                         <10 ms jitter (target)
```

## Important Files

| File | Purpose |
|------|---------|
| `src/bitnet_tmac_lut.mojo` | Main inference with T-MAC LUT |
| `src/kernels/tmac_kernel.c` | C FFI kernel (AVX2/NEON) |
| `src/kernels/cuda/tmac_kernel.cu` | CUDA kernel (GPU) |
| `src/edgellm/ffi/tmac_kernel.mojo` | Mojo FFI wrapper (CPU) |
| `src/edgellm/ffi/cuda_kernel.mojo` | Mojo FFI wrapper (CUDA) |
| `src/edgellm/ffi/kernel_selector.mojo` | Unified backend selector |
| `src/edgellm/ffi/test_ffi.mojo` | FFI integration test |
| `benchmarks/edgellm_benchmark.py` | Automated benchmark suite |
| `scripts/quantize/quantize_bitnet.py` | BitNet quantization |
| `Dockerfile.mojo` | Mojo development container |
| `Dockerfile.benchmark` | Benchmark container for fly.io |
| `PAPER_ROADMAP.md` | Research paper roadmap |
| `BENCHMARK_REPORT.md` | Benchmark comparison report |

## Architecture

### Hybrid Mojo + C/CUDA FFI

```
┌─────────────────────────────────────────────────┐
│              Mojo Layer (95%)                   │
│  • Memory management (ownership, no GC)         │
│  • Control flow, model loading                  │
│  • Transformer forward pass                     │
│  • LUT building, parallelization               │
└─────────────────────────────────────────────────┘
                      │
              Kernel Selector
          (Auto-detects best backend)
                      │
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
┌───────────┐  ┌───────────┐  ┌───────────┐
│   CUDA    │  │  AVX2/    │  │   Pure    │
│   (GPU)   │  │   NEON    │  │   Mojo    │
│           │  │   (CPU)   │  │ (Fallback)│
│ 80-400+   │  │  30-50    │  │   8-15    │
│  tok/s    │  │  tok/s    │  │   tok/s   │
└───────────┘  └───────────┘  └───────────┘
```

### Fine-Tuning → Deploy Pipeline

```
HuggingFace Model → QLoRA Fine-tune → Merge → Quantize → T-MAC → Deploy
                    (FREE Colab)              (BitNet)   (.tmac2)
```

## Build Commands

```bash
# Docker development (Intel Mac or Linux)
docker compose -f docker-compose.mojo.yml up --build

# C kernel build (x86)
cd src/kernels && make

# C kernel build (ARM)
cd src/kernels && make

# CUDA kernel build
cd src/kernels && make cuda

# CUDA kernel build (Jetson Nano)
cd src/kernels && make cuda-jetson

# CUDA kernel build (RTX)
cd src/kernels && make cuda-rtx

# Build all (CPU + GPU)
cd src/kernels && make all-gpu

# Mojo inference (in Docker)
pixi run mojo build -O3 src/bitnet_tmac_lut.mojo -o bin/edgellm

# Run benchmarks
python benchmarks/edgellm_benchmark.py --compare --runs 100
```

## Quantization Commands

```bash
# Quantize SmolLM-135M to BitNet format
python scripts/quantize/quantize_bitnet.py \
    --model HuggingFaceTB/SmolLM-135M \
    --output models/smollm-135m.tmac2.bin

# Verify quantization
python scripts/quantize/verify_bitnet.py models/smollm-135m.tmac2.bin
```

## Benchmark Commands

```bash
# Full comparison (EdgeLLM vs Ollama)
python benchmarks/edgellm_benchmark.py --compare --runs 100 -o results.json

# EdgeLLM only
python benchmarks/edgellm_benchmark.py --backend edgellm --model models/smollm-135m.tmac2.bin

# Ollama only
python benchmarks/edgellm_benchmark.py --backend ollama --model smollm:135m
```

## Mojo FFI Integration

### Working Pattern (OwnedDLHandle)
```mojo
from sys.ffi import OwnedDLHandle
from memory import UnsafePointer

fn main() raises:
    var handle = OwnedDLHandle("/path/to/libtmac_kernel.so")

    # Call C functions
    var features = handle.call["get_cpu_features", Int32]()

    # With pointers
    var input_data = List[Float32]()
    var input_ptr = input_data.unsafe_ptr()
    handle.call["rmsnorm_avx2", NoneType](output_ptr, input_ptr, weight_ptr, size, eps)
```

### Known Mojo API Changes
- `DLHandle` → `OwnedDLHandle`
- `UnsafePointer.alloc()` → `List[T]().unsafe_ptr()`
- `list.data` → `list.unsafe_ptr()`

## Target Hardware

### CPU-Only Devices
| Device | Price | RAM | Model | Expected Speed |
|--------|-------|-----|-------|----------------|
| Pi Zero 2 W | **$15** | 512MB | SmolLM-135M | 5-10 tok/s |
| Pi 4 | $35 | 4GB | Qwen-0.5B | 8-15 tok/s |
| Pi 5 | $80 | 8GB | Llama-1B | 20-40 tok/s |

### GPU-Accelerated Devices
| Device | Price | GPU | Model | Expected Speed |
|--------|-------|-----|-------|----------------|
| Jetson Nano | $99 | Maxwell 128 CUDA | SmolLM-135M | **80-120 tok/s** |
| Jetson Orin | $499 | Ampere 1024 CUDA | Llama-1B | **200-400 tok/s** |
| RTX 3090 | $1500 | Ampere 10496 CUDA | Llama-3B | **400-600 tok/s** |
| RTX 4090 | $1999 | Ada 16384 CUDA | Llama-7B | **600-1000 tok/s** |

## Platform Support

| Platform | Mojo Native | Docker | CUDA | Status |
|----------|-------------|--------|------|--------|
| Linux x86_64 | Yes | Yes | Yes | Full support |
| Linux ARM64 (Jetson) | Yes | Yes | Yes | Full support |
| macOS ARM64 | Yes | Yes | No | CPU only |
| macOS x86_64 | No | Yes | No | Docker only |
| ARM64 (Pi) | Coming | Yes | No | CPU only |

## References

- [T-MAC Paper](https://arxiv.org/abs/2407.00088) - EuroSys 2025
- [BitNet Paper](https://arxiv.org/abs/2402.17764) - 1.58-bit LLMs
- [NoMAD-Attention](https://arxiv.org/abs/2403.01273) - NeurIPS 2024
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Efficient fine-tuning
- [Mojo FFI Docs](https://docs.modular.com/mojo/stdlib/sys/ffi/)

## Research Findings

### Key Insights
- LLM inference is memory-bound, not compute-bound
- BitNet 1.58-bit: 4.8x compression vs FP16
- Mojo: 0ms GC pauses (deterministic latency)
- T-MAC eliminates multiplications via lookup tables

### Competitive Positioning
- Ollama: Higher throughput, but variable latency
- EdgeLLM: Lower throughput, but deterministic latency + smaller models
- Use case: Real-time robotics, voice assistants, IoT automation
