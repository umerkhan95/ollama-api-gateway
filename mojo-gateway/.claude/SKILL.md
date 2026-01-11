# EdgeLLM CUDA Optimization Roadmap

## Current Performance (Jan 11, 2026)

### Benchmark Results on Tesla T4 GPU

| Metric | EdgeLLM CUDA | Ollama | Gap |
|--------|--------------|--------|-----|
| Throughput | 34.6 tok/s | 423.3 tok/s | **12x slower** |
| Latency | 28.87 ms/tok | 2.36 ms/tok | 12x higher |
| Jitter | ±1.0 tok/s | ±41.9 tok/s | **EdgeLLM wins** |

---

## Root Cause Analysis

### Critical Bottleneck Identified

Looking at `src/kernels/cuda/tmac_kernel.cu:396-436`:

```c
int tmac_matmul_cuda(...) {
    // PROBLEM: Every single call does these transfers
    CUDA_CHECK(cudaMemcpy(d_weights, weights, ..., cudaMemcpyHostToDevice));     // ~1ms
    CUDA_CHECK(cudaMemcpy(d_activations, activations, ..., cudaMemcpyHostToDevice)); // ~0.5ms
    CUDA_CHECK(cudaMemcpy(d_scales, scales, ..., cudaMemcpyHostToDevice));       // ~0.1ms

    // Kernel execution: ~0.1ms (actually fast!)
    tmac_matmul_kernel<<<grid, block>>>(...);

    CUDA_CHECK(cudaMemcpy(output, d_output, ..., cudaMemcpyDeviceToHost));       // ~0.5ms
    return 0;
}
```

**Per-call overhead: ~2.1ms** (mostly memory transfers)
**Per-token calls: ~15-20** (RMSNorm, QKV, attention, FFN, etc.)
**Total overhead: ~30-40ms per token** (vs 2.36ms for Ollama)

### Why Ollama is 12x Faster

Ollama (via llama.cpp) keeps all data GPU-resident:
1. Model weights loaded to GPU once at startup
2. Activations stay on GPU between layers
3. Only final token logits transferred back to CPU
4. Uses CUDA Graphs to replay entire forward pass

---

## Research-Backed Optimization Strategies

### Academic References

1. **[FlashInfer](https://arxiv.org/abs/2501.01005)** (MLSys 2025)
   - Block-sparse KV-cache format
   - 28-30% latency reduction for long contexts
   - JIT compilation for customizable attention

2. **[Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)** (Stanford CRFM)
   - Parallelizes over K/V sequence length
   - Maximizes GPU utilization for batch_size=1
   - Critical for inference (vs training)

3. **[FlashAttention-2](https://arxiv.org/abs/2307.08691)** (Dao et al.)
   - Tiled attention with fused softmax
   - O(N) memory instead of O(N²)
   - Up to 22% faster than cuDNN attention

4. **[BitNet b1.58](https://arxiv.org/abs/2504.12285)** (Microsoft)
   - Native 1-bit inference optimizations
   - bitnet.cpp achieves 2.37x-6.17x speedup on x86 CPUs
   - GPU hardware not yet optimized for 1-bit (opportunity!)

### Key Insight from FlashDecoding

```
Problem: During inference, query length = 1
         If batch_size < num_SMs (40 on T4), GPU underutilized

Solution: Parallelize over K/V sequence length instead
          - Split K/V into blocks
          - Each SM processes one K/V block
          - Final reduction combines partial results
```

---

## Implementation Plan

### Phase 1: Eliminate Memory Transfer Overhead
**Target: 150+ tok/s | Effort: Medium | Impact: 5x**

**The Fix**: Keep all data GPU-resident.

```c
// NEW API: Persistent GPU memory
int cuda_load_model(const char* model_path);  // Load weights to GPU once
int cuda_forward(int token_id);               // Everything stays on GPU
int cuda_get_logits(float* output);           // Only transfer final output
```

**Changes Required**:

1. **New `cuda_load_model()` function**
   - Parse .tmac2 file
   - Allocate all weight tensors on GPU
   - Keep handles in static/global state

2. **Fused forward pass**
   - Single CUDA stream for entire forward
   - No intermediate HOST↔DEVICE transfers
   - Only activations allocated per-inference

3. **CUDA streams for async**
   - Overlap computation with minimal I/O
   - Pipeline token preparation

**Estimated Performance**:
- Memory overhead eliminated: -2ms/call × 20 calls = **-40ms/token**
- Kernel execution only: ~3-5ms/token
- **Expected: 150-200 tok/s**

---

### Phase 2: Mojo-Native GPU Implementation
**Target: 250+ tok/s | Effort: High | Impact: 2x over Phase 1**

Mojo now has native GPU support (no C FFI needed):
- [Mojo GPU Tutorial](https://docs.modular.com/mojo/manual/gpu/intro-tutorial/)
- [Mojo GPU Fundamentals](https://docs.modular.com/mojo/manual/gpu/fundamentals/)

**Benefits over C FFI**:
- No Python ctypes overhead
- Unified memory model (host/device in same language)
- Better compiler optimizations
- Works on NVIDIA, AMD, and Apple Metal

**Implementation**:

```mojo
from gpu import thread_idx, block_idx, block_dim, DeviceContext, DeviceBuffer

# Direct GPU kernel in Mojo
fn rmsnorm_kernel(
    output: DeviceBuffer[DType.float32],
    input: DeviceBuffer[DType.float32],
    weight: DeviceBuffer[DType.float32],
    size: Int,
    eps: Float32
):
    var tid = thread_idx.x
    var idx_base = block_idx.x * size

    # Shared memory for reduction
    var smem = __shared__[Float32, 256]()

    # Compute sum of squares (same algorithm as C)
    var sum_sq: Float32 = 0.0
    for i in range(tid, size, block_dim.x):
        var val = input[idx_base + i]
        sum_sq += val * val

    # Warp reduction
    sum_sq = warp_reduce_sum(sum_sq)
    # ... rest of kernel
```

**Migration Path**:
1. Port `rmsnorm_kernel` to Mojo (simplest)
2. Benchmark vs C version
3. If faster, port remaining kernels
4. If not, stay with C FFI + persistent memory

---

### Phase 3: Kernel Fusion (FlashDecoding Style)
**Target: 400+ tok/s | Effort: High | Impact: 2x**

Based on [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html):

**Fused Operations**:

1. **FusedRMSNorm+QKV**
   ```
   Before: rmsnorm() → linear_q() → linear_k() → linear_v() [4 kernels]
   After:  fused_rmsnorm_qkv() [1 kernel]
   ```

2. **FusedAttention (FlashDecoding)**
   ```
   Before: matmul(Q,K) → softmax() → matmul(,V) [3 kernels + O(N²) memory]
   After:  flash_attention() [1 kernel, O(N) memory]
   ```

3. **FusedFFN**
   ```
   Before: gate_proj() → up_proj() → silu() → mul() → down_proj() [5 kernels]
   After:  fused_ffn() [1 kernel]
   ```

**Kernel Launch Reduction**:
- Current: ~20+ kernel launches per token
- After fusion: ~5 kernel launches per token
- Saved overhead: ~10ms/token

---

### Phase 4: T-MAC Optimization for Turing Architecture
**Target: 600+ tok/s | Effort: Very High | Impact: 1.5x**

Tesla T4 has Tensor Cores (INT8) that we're not using.

**Current T-MAC approach** (naive):
```cuda
// Line 101-102 in tmac_kernel.cu
if (ternary != 0) {
    atomicAdd(&lut[act_idx][col_off], ternary * act);  // SLOW: atomic
}
```

**Optimized T-MAC** (tensor core aware):
```cuda
// Use INT8 tensor cores for ternary multiplication
// Pack multiple ternary values for WMMA operations
// Avoid atomics with privatized LUTs per warp
```

**Key Optimizations**:
1. Remove `atomicAdd` - use warp-private LUTs
2. Use shared memory banks efficiently (avoid conflicts)
3. Coalesced global memory access patterns
4. INT8 tensor cores for ternary→INT8 matmul equivalent

---

## Benchmark Milestones

### Milestone 1: 100 tok/s (Memory Fix)
- [ ] Implement `cuda_load_model()` with persistent GPU weights
- [ ] Modify forward pass to avoid HOST↔DEVICE transfers
- [ ] Benchmark: `python benchmarks/edgellm_benchmark.py --compare`

### Milestone 2: 200 tok/s (Mojo GPU or Optimized C)
- [ ] Either: Port kernels to Mojo native GPU
- [ ] Or: Optimize C kernels with CUDA streams
- [ ] Benchmark on T4

### Milestone 3: 400 tok/s (Kernel Fusion)
- [ ] Implement FusedRMSNorm+QKV
- [ ] Implement FlashDecoding-style attention
- [ ] Reduce kernel launches to <5/token

### Milestone 4: 600 tok/s (Beat Ollama)
- [ ] T-MAC tensor core optimization
- [ ] CUDA Graph for full forward pass
- [ ] Profile with Nsight Systems

---

## Quick Win: Persistent Memory

**Immediate action** - modify `tmac_matmul_cuda` in `tmac_kernel.cu`:

```c
// ADD: Flag to skip weight transfer if already loaded
static int weights_loaded = 0;

int tmac_matmul_cuda_persistent(
    const int8_t* weights,  // Can be NULL if weights_loaded
    const float* activations,
    float* output,
    const float* scales,
    int M, int N, int K
) {
    // Only transfer weights once
    if (!weights_loaded && weights != NULL) {
        CUDA_CHECK(cudaMemcpy(d_weights, weights, weight_bytes, cudaMemcpyHostToDevice));
        weights_loaded = 1;
    }

    // Transfer only activations (small)
    CUDA_CHECK(cudaMemcpy(d_activations, activations, act_size * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel (fast)
    tmac_matmul_kernel<<<grid, block>>>(...);

    // Transfer only output (small)
    CUDA_CHECK(cudaMemcpy(output, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}
```

**Expected gain**: 2-3x immediately (weights ~80% of transfer time)

---

## Alternative Strategy: Edge Device Positioning

If matching datacenter GPU performance proves difficult, pivot to edge advantage:

| Metric | EdgeLLM Advantage | Ollama Weakness |
|--------|-------------------|-----------------|
| Memory | 53MB model (BitNet) | 91MB+ model |
| Minimum RAM | 512MB | 4GB+ |
| Latency jitter | <10ms | 5566ms measured |
| $15 hardware | Runs | Cannot run |

**Target markets where EdgeLLM wins**:
- Robotics (deterministic latency required)
- IoT/embedded (memory constrained)
- Privacy-sensitive (offline operation)
- Battery-powered (lower energy per token)

---

## Sources

- [FlashInfer (MLSys 2025)](https://arxiv.org/abs/2501.01005)
- [Flash-Decoding (Stanford CRFM)](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [BitNet b1.58 Technical Report](https://arxiv.org/abs/2504.12285)
- [Mojo GPU Tutorial](https://docs.modular.com/mojo/manual/gpu/intro-tutorial/)
- [Mojo GPU Fundamentals](https://docs.modular.com/mojo/manual/gpu/fundamentals/)
- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
