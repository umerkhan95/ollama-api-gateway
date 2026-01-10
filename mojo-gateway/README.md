# Mojo Gateway - High-Performance LLM Inference Engine

A high-performance LLM inference engine written in **Mojo**, featuring T-MAC (Table Lookup-based) inference for extreme memory efficiency.

## Overview

This project provides multiple LLM inference implementations optimized for different use cases:

- **T-MAC Inference** - 16x memory compression using lookup tables (no multiplication!)
- **SIMD Optimized** - Vectorized operations for maximum throughput
- **Int4 Quantized** - 7x memory reduction with good quality
- **API Gateway** - Production-ready HTTP server with auth and rate limiting

### Performance Highlights

| Implementation | Speed | Memory | Compression |
|---------------|-------|--------|-------------|
| Float32 SIMD | 29 tok/s | 418 MB | 1x |
| Int4 Quantized | 13 tok/s | 62 MB | 7.1x |
| **T-MAC v2** | **17 tok/s** | **27 MB** | **15.7x** |

**T-MAC enables running 7x larger models in the same memory footprint.**

## T-MAC: Multiplication-Free Inference

T-MAC eliminates multiplication in matrix operations using precomputed lookup tables:

```
Traditional: output[i] = Σ weight[i,j] × activation[j]  // Expensive
T-MAC:       output[i] = Σ LUT[group, weight_pattern]   // Just lookups!
```

### Quick Start with T-MAC

```bash
# Quantize model to T-MAC format
python scripts/quantize_tmac_v2.py model.bin model.tmac2.bin

# Build inference engine
mojo build -O3 src/llama2_tmac_v2.mojo -o llama2_tmac

# Run inference
./llama2_tmac model.tmac2.bin -z tokenizer.bin -n 128 -t 0.8
```

See [README_TMAC.md](README_TMAC.md) for detailed T-MAC documentation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Mojo API Gateway                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ HTTP Server │  │ Auth/JWT    │  │ Rate Limiter        │ │
│  │ (Lightbug)  │  │ (Mojo)      │  │ (SIMD-accelerated)  │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                           │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Request Router & Handlers                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 MAX Engine                           │   │
│  │    (In-process LLM inference, GPU-accelerated)       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Core Features
- **Pure Mojo HTTP Server** - Using Lightbug HTTP framework
- **MAX Engine Integration** - Native LLM inference without Python overhead
- **SIMD-Accelerated Statistics** - Vectorized metrics computation
- **Sliding Window Rate Limiting** - Efficient request throttling
- **JWT Authentication** - Secure API key validation
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI clients

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information |
| `/health` | GET | Health check |
| `/ready` | GET | Kubernetes readiness probe |
| `/live` | GET | Kubernetes liveness probe |
| `/api/generate` | POST | Text generation |
| `/api/chat` | POST | Chat completions |
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/api/models` | GET | List available models |
| `/v1/models` | GET | OpenAI-compatible models list |
| `/api/keys` | POST | Create API key (admin) |
| `/api/keys` | GET | List API keys (admin) |
| `/api/keys/{id}` | DELETE | Revoke API key (admin) |
| `/api/stats` | GET | User statistics |
| `/api/stats/detailed` | GET | Detailed statistics |
| `/api/admin/stats` | GET | Admin statistics |

## Getting Started

### Prerequisites

1. **Install Modular CLI (Magic)**:
   ```bash
   curl -ssL https://magic.modular.com | bash
   ```

2. **Install MAX and Mojo**:
   ```bash
   magic install max mojo
   ```

### Installation

1. **Clone and navigate to the Mojo gateway**:
   ```bash
   cd mojo-gateway
   ```

2. **Install dependencies**:
   ```bash
   magic install
   ```

3. **Run in development mode**:
   ```bash
   magic run dev
   ```

4. **Build optimized binary**:
   ```bash
   magic run build
   ./bin/gateway
   ```

### Docker Deployment

**CPU-only deployment**:
```bash
docker-compose up mojo-gateway
```

**GPU deployment (NVIDIA)**:
```bash
docker-compose --profile gpu up mojo-gateway-gpu
```

**Full stack with monitoring**:
```bash
docker-compose --profile monitoring up
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GATEWAY_HOST` | `0.0.0.0` | Host to bind |
| `GATEWAY_PORT` | `8080` | Port to bind |
| `JWT_SECRET` | (required) | Secret for JWT signing |
| `MODEL_PATH` | `meta-llama/Llama-3.1-8B-Instruct` | Model path or HuggingFace ID |
| `LOG_LEVEL` | `INFO` | Logging level |

## Usage Examples

### Text Generation

```bash
curl -X POST http://localhost:8080/api/generate \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b-instruct",
    "prompt": "Write a haiku about programming",
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### Chat Completion

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b-instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### OpenAI-Compatible Request

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="YOUR_API_KEY"
)

response = client.chat.completions.create(
    model="llama-3.1-8b-instruct",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

## Project Structure

```
mojo-gateway/
├── README.md                  # This file
├── README_TMAC.md             # T-MAC detailed documentation
├── src/
│   ├── llama2_tmac.mojo       # T-MAC v1: Pure ternary inference
│   ├── llama2_tmac_v2.mojo    # T-MAC v2: Scaled ternary (better quality)
│   ├── llama2_parallel.mojo   # Float32 SIMD parallel inference
│   ├── llama2_simd.mojo       # Float32 SIMD inference
│   ├── llama2_int4.mojo       # Int4 quantized inference
│   ├── llama2.mojo            # Basic inference
│   ├── main.mojo              # API Gateway entry point
│   ├── auth/                  # Authentication modules
│   ├── middleware/            # Rate limiting, logging
│   └── utils/                 # Utilities
├── scripts/
│   ├── quantize_tmac.py       # T-MAC v1 quantization
│   ├── quantize_tmac_v2.py    # T-MAC v2 quantization (with scales)
│   ├── quantize_int4.py       # Int4 quantization
│   └── benchmark_all.sh       # Benchmark script
└── inference/                 # Additional inference utilities
```

## Performance Comparison

| Metric | Python (FastAPI) | Mojo Gateway | Improvement |
|--------|------------------|--------------|-------------|
| Gateway overhead | ~50ms | ~5ms | 10x |
| First token latency | Variable | 70% faster | 3x |
| Memory usage | High (GC) | Low (no GC) | 30-50% |
| Throughput | GIL-limited | True parallel | 2-5x |

## Roadmap

### Completed
- [x] T-MAC lookup table inference (16x compression)
- [x] Int4 quantization (7x compression)
- [x] SIMD vectorized inference
- [x] Parallel multi-core inference
- [x] Per-row scaled ternary (T-MAC v2)

### In Progress
- [ ] GPTQ-style calibration for better quality
- [ ] Quantization-aware fine-tuning
- [ ] Training scripts for ternary models

### Planned
- [ ] Streaming responses (SSE)
- [ ] SIMD-optimized LUT (TBL/PSHUF instructions)
- [ ] Batch inference
- [ ] WebSocket support

## Contributing

Contributions are welcome! Please read the contribution guidelines before submitting a pull request.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Modular](https://www.modular.com/) - Mojo language and MAX Engine
- [T-MAC Paper](https://arxiv.org/abs/2407.00088) - Table lookup inference algorithm
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) - Ternary weight training research
- [llama2.c](https://github.com/karpathy/llama2.c) - Reference implementation
- [Lightbug](https://github.com/Lightbug-HQ/lightbug_http) - Mojo HTTP framework
