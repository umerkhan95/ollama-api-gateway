# EdgeLLM CUDA Optimization Roadmap

## Current Performance (Jan 11, 2026)

### Phase 1 Results: Persistent GPU Memory ✅ COMPLETE

| Metric | Original API | Persistent API | Ollama | vs Ollama |
|--------|--------------|----------------|--------|-----------|
| Throughput | 75.5 tok/s | **296.4 tok/s** | 423.3 tok/s | 1.4x gap |
| Latency | 13.2 ms/tok | **3.4 ms/tok** | 2.36 ms/tok | Close! |
| Jitter | ±1.0 tok/s | ±1.0 tok/s | ±41.9 tok/s | **EdgeLLM wins** |

**Phase 1 Achievement: 3.93x speedup** (from 75.5 → 296.4 tok/s)

### Layer-by-Layer Benchmark (Tesla T4)

| Layer | Original (ms) | Persistent (ms) | Speedup |
|-------|---------------|-----------------|---------|
| QKV Projection | 1.074 | 0.982 | 1.09x |
| Output Projection | 0.419 | 0.175 | **2.39x** |
| FFN Up | 0.480 | 0.401 | 1.20x |
| FFN Down | 0.492 | 0.391 | 1.26x |
| Embedding Lookup | 11.316 | 9.798 | 1.15x |

### Memory Transfer Reduction

```
FFN Up Layer [1536, 576]:
  Original API:   230.2 KB per call
  Persistent API:   8.2 KB per call
  Reduction:      96.4%
```

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

### Milestone 1: 150 tok/s (Memory Fix) ✅ ACHIEVED: 296 tok/s
- [x] Implement `cuda_load_weights()` with persistent GPU weights
- [x] Implement `tmac_matmul_cuda_persistent()` API
- [x] Implement `rmsnorm_cuda_persistent()` API
- [x] Benchmark: 3.93x speedup verified on T4

### Milestone 2: 400 tok/s (Kernel Fusion) 🚧 IN PROGRESS
- [ ] Implement FusedRMSNorm+MatMul kernel
- [ ] Add CUDA streams for async H2D/D2H transfers
- [ ] Reduce kernel launches per token
- [ ] Target: Close remaining 1.4x gap with Ollama

### Milestone 3: 500+ tok/s (Beat Ollama)
- [ ] T-MAC tensor core optimization (INT8 WMMA)
- [ ] CUDA Graph for full forward pass
- [ ] Profile with Nsight Systems
- [ ] Warp-private LUTs (remove atomicAdd)

---

## Phase 1 Implementation ✅ COMPLETE

**Implemented in `tmac_kernel.cu`**:

```c
// Persistent weight storage (GPU-resident)
static int8_t* d_persistent_weights = nullptr;
static float* d_persistent_scales = nullptr;
static int weights_on_gpu = 0;

// Load weights once at model init
int cuda_load_weights(const int8_t* weights, const float* scales,
                      int weight_bytes, int num_rows) {
    CUDA_CHECK(cudaMalloc(&d_persistent_weights, weight_bytes));
    CUDA_CHECK(cudaMemcpy(d_persistent_weights, weights, weight_bytes, H2D));
    weights_on_gpu = 1;
    return 0;
}

// Fast inference - weights already on GPU
int tmac_matmul_cuda_persistent(const float* activations, float* output,
                                 int M, int N, int K) {
    // Transfer only activations (small)
    CUDA_CHECK(cudaMemcpy(d_activations, activations, act_size, H2D));

    // Kernel uses d_persistent_weights (already on GPU!)
    tmac_matmul_kernel<<<grid, block>>>(d_persistent_weights, ...);

    // Transfer only output (small)
    CUDA_CHECK(cudaMemcpy(output, d_output, out_size, D2H));
    return 0;
}
```

**Result**: 3.93x speedup (exceeded 2-3x target)

---

## Phase 2 Implementation 🚧 IN PROGRESS

### Strategy: Kernel Fusion + CUDA Streams

**Current bottleneck analysis** (at 296 tok/s):
- Per-token time: 3.4ms
- Kernel launches: ~6 per layer × 9 layers = 54 launches
- Launch overhead: ~5-10μs per launch = ~0.5ms total
- Remaining H2D/D2H: ~0.5ms per layer

**Phase 2 Optimizations**:

#### 1. FusedRMSNorm+MatMul Kernel
```cuda
// BEFORE: 2 kernel launches + intermediate buffer
rmsnorm_kernel<<<...>>>(norm_out, input, weights);
tmac_matmul_kernel<<<...>>>(output, norm_out, weights);

// AFTER: 1 kernel launch, no intermediate buffer
fused_rmsnorm_matmul_kernel<<<...>>>(output, input, norm_weights, matmul_weights);
```

Benefits:
- 50% fewer kernel launches for attention/FFN blocks
- Eliminates intermediate buffer allocation
- Better GPU occupancy

#### 2. CUDA Streams for Async Transfers
```cuda
cudaStream_t compute_stream, transfer_stream;

// Overlap: Transfer next layer's activations while computing current layer
cudaMemcpyAsync(d_next_act, h_next_act, size, H2D, transfer_stream);
kernel<<<..., compute_stream>>>(d_curr_act, ...);
cudaStreamSynchronize(compute_stream);
```

#### 3. Pinned Memory for Faster Transfers
```cuda
// Use pinned (page-locked) memory for 2x faster H2D/D2H
cudaMallocHost(&h_activations, size);  // Pinned on host
cudaMemcpyAsync(d_act, h_activations, size, H2D, stream);  // Faster!
```

**Target**: 400+ tok/s (closing the 1.4x gap with Ollama)

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
