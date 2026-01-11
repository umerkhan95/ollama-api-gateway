/**
 * EdgeLLM CUDA T-MAC Kernels
 *
 * High-performance GPU implementation of Table-lookup Matrix Multiplication (T-MAC)
 * for BitNet 1.58-bit quantized models.
 *
 * Key optimizations:
 * - Lookup tables in shared memory (fast access)
 * - Coalesced global memory reads
 * - Warp-level reductions
 * - Minimal thread divergence
 *
 * Target: NVIDIA Jetson Nano/Orin, RTX GPUs
 */

#include "tmac_kernel_cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Configuration
#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define LUT_SIZE 16          // 4-bit activation indices (0-15)
#define TILE_K 128           // K dimension tile size
#define TILE_N 64            // N dimension tile size

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

/**
 * T-MAC Lookup Table Kernel
 *
 * Instead of: y = W * x (matrix multiplication)
 * We do:      y = LUT[packed_weights] (table lookup)
 *
 * The LUT is precomputed: LUT[i] = sum of activations where ternary weight pattern = i
 */
__global__ void tmac_matmul_kernel(
    const int8_t* __restrict__ packed_weights,  // Packed ternary weights (2 bits each)
    const float* __restrict__ activations,       // Input activations
    float* __restrict__ output,                  // Output buffer
    const float* __restrict__ scales,            // Per-row scales
    int M,                                       // Output rows
    int N,                                       // Output cols (batch)
    int K                                        // Inner dimension
) {
    // Shared memory for lookup tables (one per output row in block)
    __shared__ float lut[LUT_SIZE][TILE_N];

    int row = blockIdx.x;           // Output row
    int col_base = blockIdx.y * TILE_N;  // Output column base
    int tid = threadIdx.x;

    if (row >= M) return;

    // Initialize LUT to zero
    for (int i = tid; i < LUT_SIZE * TILE_N; i += BLOCK_SIZE) {
        int lut_idx = i / TILE_N;
        int col_off = i % TILE_N;
        lut[lut_idx][col_off] = 0.0f;
    }
    __syncthreads();

    // Build lookup table for this row
    // For 4-bit activation quantization, we have 16 possible values
    // LUT[i] = sum of activations where the 4-bit index equals i
    float scale = scales[row];

    // Process K dimension in tiles
    for (int k_base = 0; k_base < K; k_base += TILE_K) {
        int k_end = min(k_base + TILE_K, K);

        // Each thread processes multiple K elements
        for (int k = k_base + tid; k < k_end; k += BLOCK_SIZE) {
            // Get packed weight (4 ternary values per byte)
            int weight_idx = row * ((K + 3) / 4) + (k / 4);
            int8_t packed = packed_weights[weight_idx];

            // Extract ternary value for this k (-1, 0, +1)
            int shift = (k % 4) * 2;
            int ternary = ((packed >> shift) & 0x3) - 1;  // Map 0,1,2 to -1,0,1

            // Update LUT based on activation quantization
            for (int col_off = 0; col_off < TILE_N && (col_base + col_off) < N; col_off++) {
                float act = activations[k * N + col_base + col_off];

                // Quantize activation to 4-bit index
                int act_idx = min(15, max(0, (int)((act + 1.0f) * 8.0f)));

                // Accumulate based on ternary weight
                if (ternary != 0) {
                    atomicAdd(&lut[act_idx][col_off], ternary * act);
                }
            }
        }
    }
    __syncthreads();

    // Reduce LUT to final output
    for (int col_off = tid; col_off < TILE_N && (col_base + col_off) < N; col_off += BLOCK_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < LUT_SIZE; i++) {
            sum += lut[i][col_off];
        }
        output[row * N + col_base + col_off] = sum * scale;
    }
}

/**
 * Optimized T-MAC kernel using warp-level primitives
 * Better memory coalescing and reduced atomic operations
 */
__global__ void tmac_matmul_kernel_v2(
    const uint8_t* __restrict__ packed_weights,  // Packed ternary weights
    const half* __restrict__ activations,        // FP16 activations
    half* __restrict__ output,                   // FP16 output
    const half* __restrict__ scales,             // Per-row scales
    int M,                                       // Output rows
    int N,                                       // Output cols
    int K                                        // Inner dimension
) {
    // Shared memory for partial sums
    __shared__ float smem[BLOCK_SIZE];

    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    float scale = __half2float(scales[row]);

    // Each thread accumulates over K
    int k_per_byte = 4;  // 4 ternary values per byte
    int weight_row_bytes = (K + k_per_byte - 1) / k_per_byte;

    for (int k = tid; k < K; k += BLOCK_SIZE) {
        // Load activation
        float act = __half2float(activations[k * N + col]);

        // Load and unpack weight
        int byte_idx = k / k_per_byte;
        int bit_offset = (k % k_per_byte) * 2;
        uint8_t packed = packed_weights[row * weight_row_bytes + byte_idx];
        int ternary = ((packed >> bit_offset) & 0x3) - 1;  // -1, 0, +1

        // Accumulate
        sum += ternary * act;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread in each warp writes to shared memory
    if (lane == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction across warps (first warp only)
    if (warp_id == 0) {
        sum = (tid < (BLOCK_SIZE / WARP_SIZE)) ? smem[tid] : 0.0f;

        #pragma unroll
        for (int offset = (BLOCK_SIZE / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (tid == 0) {
            output[row * N + col] = __float2half(sum * scale);
        }
    }
}

/**
 * RMSNorm CUDA kernel
 */
__global__ void rmsnorm_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int size,
    float eps
) {
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx_base = blockIdx.x * size;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        float val = input[idx_base + i];
        sum_sq += val * val;
    }

    // Reduce within block
    smem[tid] = sum_sq;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    // Compute normalization factor
    float rms = rsqrtf(smem[0] / size + eps);

    // Apply normalization
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        output[idx_base + i] = input[idx_base + i] * rms * weight[i];
    }
}

/**
 * Softmax CUDA kernel
 */
__global__ void softmax_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int size
) {
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx_base = blockIdx.x * size;

    // Find max
    float max_val = -INFINITY;
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        max_val = fmaxf(max_val, input[idx_base + i]);
    }

    smem[tid] = max_val;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        float exp_val = expf(input[idx_base + i] - max_val);
        output[idx_base + i] = exp_val;
        sum += exp_val;
    }

    smem[tid] = sum;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    sum = smem[0];

    // Normalize
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        output[idx_base + i] /= sum;
    }
}

/**
 * RoPE (Rotary Position Embedding) CUDA kernel
 */
__global__ void rope_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ freq_cis_real,
    const float* __restrict__ freq_cis_imag,
    int head_dim,
    int pos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;

    if (idx >= half_dim) return;

    float cos_val = freq_cis_real[pos * half_dim + idx];
    float sin_val = freq_cis_imag[pos * half_dim + idx];

    // Rotate Q
    float q0 = q[idx * 2];
    float q1 = q[idx * 2 + 1];
    q[idx * 2] = q0 * cos_val - q1 * sin_val;
    q[idx * 2 + 1] = q0 * sin_val + q1 * cos_val;

    // Rotate K
    float k0 = k[idx * 2];
    float k1 = k[idx * 2 + 1];
    k[idx * 2] = k0 * cos_val - k1 * sin_val;
    k[idx * 2 + 1] = k0 * sin_val + k1 * cos_val;
}

// ============================================================================
// C Interface for Mojo FFI
// ============================================================================

extern "C" {

// Device memory handles
static float* d_weights = nullptr;
static float* d_activations = nullptr;
static float* d_output = nullptr;
static float* d_scales = nullptr;
static int cuda_initialized = 0;

/**
 * Initialize CUDA context and allocate device memory
 */
int cuda_init(int max_weights_bytes, int max_activations, int max_output) {
    if (cuda_initialized) return 0;

    // Check for CUDA devices
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    }

    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("EdgeLLM CUDA Engine initialized on: %s\n", prop.name);
    printf("  - Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  - Total memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("  - SM count: %d\n", prop.multiProcessorCount);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_weights, max_weights_bytes));
    CUDA_CHECK(cudaMalloc(&d_activations, max_activations * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, max_output * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales, max_output * sizeof(float)));

    cuda_initialized = 1;
    return 0;
}

/**
 * Cleanup CUDA resources
 */
void cuda_cleanup() {
    if (!cuda_initialized) return;

    cudaFree(d_weights);
    cudaFree(d_activations);
    cudaFree(d_output);
    cudaFree(d_scales);

    d_weights = nullptr;
    d_activations = nullptr;
    d_output = nullptr;
    d_scales = nullptr;

    cuda_initialized = 0;
}

/**
 * T-MAC Matrix Multiplication (GPU)
 *
 * @param weights     Packed ternary weights (host)
 * @param activations Input activations (host)
 * @param output      Output buffer (host)
 * @param scales      Per-row scales (host)
 * @param M           Output rows
 * @param N           Output columns
 * @param K           Inner dimension
 */
int tmac_matmul_cuda(
    const int8_t* weights,
    const float* activations,
    float* output,
    const float* scales,
    int M, int N, int K
) {
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized. Call cuda_init() first.\n");
        return -1;
    }

    // Calculate sizes
    int weight_bytes = M * ((K + 3) / 4);
    int act_size = K * N;
    int out_size = M * N;

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_weights, weights, weight_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_activations, activations, act_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, scales, M * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 grid(M, (N + TILE_N - 1) / TILE_N);
    dim3 block(BLOCK_SIZE);

    tmac_matmul_kernel<<<grid, block>>>(
        (int8_t*)d_weights,
        d_activations,
        d_output,
        d_scales,
        M, N, K
    );

    CUDA_CHECK(cudaGetLastError());

    // Copy back to host
    CUDA_CHECK(cudaMemcpy(output, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * RMSNorm (GPU)
 */
int rmsnorm_cuda(
    float* output,
    const float* input,
    const float* weight,
    int batch_size,
    int size,
    float eps
) {
    if (!cuda_initialized) return -1;

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_activations, input, batch_size * size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, weight, size * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    rmsnorm_kernel<<<batch_size, BLOCK_SIZE>>>(
        d_output, d_activations, d_scales, size, eps
    );

    CUDA_CHECK(cudaGetLastError());

    // Copy back
    CUDA_CHECK(cudaMemcpy(output, d_output, batch_size * size * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * Softmax (GPU)
 */
int softmax_cuda(
    float* output,
    const float* input,
    int batch_size,
    int size
) {
    if (!cuda_initialized) return -1;

    CUDA_CHECK(cudaMemcpy(d_activations, input, batch_size * size * sizeof(float), cudaMemcpyHostToDevice));

    softmax_kernel<<<batch_size, BLOCK_SIZE>>>(d_output, d_activations, size);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(output, d_output, batch_size * size * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * Check if CUDA is available
 */
int cuda_available() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

/**
 * Get CUDA device name
 */
const char* cuda_device_name() {
    static char name[256];
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        strncpy(name, prop.name, 255);
        return name;
    }
    return "Unknown";
}

/**
 * Synchronize CUDA device
 */
void cuda_sync() {
    cudaDeviceSynchronize();
}

/**
 * Get CUDA device properties
 */
int cuda_device_info(
    size_t* total_memory,
    int* sm_count,
    int* compute_major,
    int* compute_minor
) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return -1;
    }

    if (total_memory) *total_memory = prop.totalGlobalMem;
    if (sm_count) *sm_count = prop.multiProcessorCount;
    if (compute_major) *compute_major = prop.major;
    if (compute_minor) *compute_minor = prop.minor;

    return 0;
}

// ============================================================================
// Phase 1: Persistent GPU Memory Implementation
// ============================================================================

// Persistent weight storage
static int8_t* d_persistent_weights = nullptr;
static float* d_persistent_scales = nullptr;
static float* d_persistent_norm_weights = nullptr;
static int persistent_weight_bytes = 0;
static int persistent_num_rows = 0;
static int persistent_norm_size = 0;
static int weights_on_gpu = 0;
static int norm_weights_on_gpu = 0;

/**
 * Load model weights to GPU memory (one-time operation)
 */
int cuda_load_weights(
    const int8_t* weights,
    const float* scales,
    int weight_bytes,
    int num_rows
) {
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized. Call cuda_init() first.\n");
        return -1;
    }

    // Free existing weights if any
    if (d_persistent_weights) {
        cudaFree(d_persistent_weights);
        d_persistent_weights = nullptr;
    }
    if (d_persistent_scales) {
        cudaFree(d_persistent_scales);
        d_persistent_scales = nullptr;
    }

    // Allocate GPU memory for weights
    CUDA_CHECK(cudaMalloc(&d_persistent_weights, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_persistent_scales, num_rows * sizeof(float)));

    // Copy weights to GPU (one-time transfer)
    CUDA_CHECK(cudaMemcpy(d_persistent_weights, weights, weight_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_persistent_scales, scales, num_rows * sizeof(float), cudaMemcpyHostToDevice));

    persistent_weight_bytes = weight_bytes;
    persistent_num_rows = num_rows;
    weights_on_gpu = 1;

    printf("EdgeLLM: Loaded %d bytes of weights to GPU (%d rows)\n", weight_bytes, num_rows);

    return 0;
}

/**
 * Unload weights from GPU memory
 */
void cuda_unload_weights() {
    if (d_persistent_weights) {
        cudaFree(d_persistent_weights);
        d_persistent_weights = nullptr;
    }
    if (d_persistent_scales) {
        cudaFree(d_persistent_scales);
        d_persistent_scales = nullptr;
    }
    persistent_weight_bytes = 0;
    persistent_num_rows = 0;
    weights_on_gpu = 0;
}

/**
 * Check if weights are loaded on GPU
 */
int cuda_weights_loaded() {
    return weights_on_gpu;
}

/**
 * T-MAC Matrix Multiplication with persistent weights
 *
 * This is the FAST version - only transfers activations, not weights.
 * Weights must be pre-loaded via cuda_load_weights().
 */
int tmac_matmul_cuda_persistent(
    const float* activations,
    float* output,
    int M, int N, int K
) {
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized. Call cuda_init() first.\n");
        return -1;
    }

    if (!weights_on_gpu) {
        fprintf(stderr, "Weights not loaded. Call cuda_load_weights() first.\n");
        return -1;
    }

    // Calculate sizes
    int act_size = K * N;
    int out_size = M * N;

    // Transfer ONLY activations to device (weights already there!)
    CUDA_CHECK(cudaMemcpy(d_activations, activations, act_size * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel using persistent weights
    dim3 grid(M, (N + TILE_N - 1) / TILE_N);
    dim3 block(BLOCK_SIZE);

    tmac_matmul_kernel<<<grid, block>>>(
        d_persistent_weights,
        d_activations,
        d_output,
        d_persistent_scales,
        M, N, K
    );

    CUDA_CHECK(cudaGetLastError());

    // Transfer output back to host
    CUDA_CHECK(cudaMemcpy(output, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

/**
 * Load normalization weights to GPU
 */
int cuda_load_norm_weights(
    const float* norm_weights,
    int size
) {
    if (!cuda_initialized) {
        fprintf(stderr, "CUDA not initialized. Call cuda_init() first.\n");
        return -1;
    }

    // Free existing norm weights if any
    if (d_persistent_norm_weights) {
        cudaFree(d_persistent_norm_weights);
        d_persistent_norm_weights = nullptr;
    }

    // Allocate and copy
    CUDA_CHECK(cudaMalloc(&d_persistent_norm_weights, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_persistent_norm_weights, norm_weights, size * sizeof(float), cudaMemcpyHostToDevice));

    persistent_norm_size = size;
    norm_weights_on_gpu = 1;

    return 0;
}

/**
 * RMSNorm using pre-loaded weights
 */
int rmsnorm_cuda_persistent(
    float* output,
    const float* input,
    int batch_size,
    int size,
    float eps
) {
    if (!cuda_initialized) return -1;
    if (!norm_weights_on_gpu) {
        fprintf(stderr, "Norm weights not loaded. Call cuda_load_norm_weights() first.\n");
        return -1;
    }

    // Copy only input to device (weights already there!)
    CUDA_CHECK(cudaMemcpy(d_activations, input, batch_size * size * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel with persistent norm weights
    rmsnorm_kernel<<<batch_size, BLOCK_SIZE>>>(
        d_output, d_activations, d_persistent_norm_weights, size, eps
    );

    CUDA_CHECK(cudaGetLastError());

    // Copy output back
    CUDA_CHECK(cudaMemcpy(output, d_output, batch_size * size * sizeof(float), cudaMemcpyDeviceToHost));

    return 0;
}

} // extern "C"
