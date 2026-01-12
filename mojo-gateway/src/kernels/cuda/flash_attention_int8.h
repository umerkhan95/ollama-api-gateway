/**
 * INT8 Tensor Core Flash Attention - Header
 *
 * Provides INT8 quantized attention using WMMA Tensor Cores.
 * ~8x speedup over FP32 on supported GPUs (sm_75+).
 */

#ifndef FLASH_ATTENTION_INT8_H
#define FLASH_ATTENTION_INT8_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize INT8 Flash Attention buffers
 *
 * @param max_batch_heads  Maximum batch * num_heads
 * @param max_cache_len    Maximum KV cache length
 * @param head_dim         Head dimension
 * @return 0 on success, -1 on failure
 */
int flash_attention_int8_init(int max_batch_heads, int max_cache_len, int head_dim);

/**
 * Cleanup INT8 Flash Attention resources
 */
void flash_attention_int8_cleanup(void);

/**
 * Quantize FP32 tensor to INT8
 *
 * @param input   Input FP32 tensor
 * @param output  Output INT8 tensor (must be pre-allocated)
 * @param scale   Output dequantization scale
 * @param size    Number of elements
 * @return 0 on success
 */
int quantize_tensor_int8(
    const float* input,
    int8_t* output,
    float* scale,
    int size
);

/**
 * Update INT8 KV cache
 *
 * @param K_int8       New key in INT8 [batch_heads, head_dim]
 * @param V_int8       New value in INT8 [batch_heads, head_dim]
 * @param scale_k      Dequantization scale for K
 * @param scale_v      Dequantization scale for V
 * @param batch_heads  Number of batch * heads
 * @param cache_pos    Position to write in cache
 * @return 0 on success
 */
int flash_attention_int8_update_cache(
    const int8_t* K_int8,
    const int8_t* V_int8,
    float scale_k,
    float scale_v,
    int batch_heads,
    int cache_pos
);

/**
 * INT8 Flash Attention Decode (with pre-quantized input)
 *
 * Single-token decode using INT8 quantized QKV.
 *
 * @param Q_int8       Query in INT8 [batch_heads, head_dim]
 * @param K_int8       New key in INT8 [batch_heads, head_dim]
 * @param V_int8       New value in INT8 [batch_heads, head_dim]
 * @param O            Output in FP32 [batch_heads, head_dim]
 * @param scale_q      Dequantization scale for Q
 * @param scale_k      Dequantization scale for K
 * @param scale_v      Dequantization scale for V
 * @param batch_heads  Number of batch * heads
 * @param cache_pos    Current cache position (0-indexed)
 * @param head_dim     Head dimension
 * @return 0 on success, -1 on failure
 */
int flash_attention_int8_decode(
    const int8_t* Q_int8,
    const int8_t* K_int8,
    const int8_t* V_int8,
    float* O,
    float scale_q,
    float scale_k,
    float scale_v,
    int batch_heads,
    int cache_pos,
    int head_dim
);

/**
 * INT8 Flash Attention Decode (with FP32 input)
 *
 * Convenience function that accepts FP32 input and handles quantization internally.
 * Slightly slower than pre-quantized version but easier to use.
 *
 * @param Q           Query in FP32 [batch_heads, head_dim]
 * @param K_new       New key in FP32 [batch_heads, head_dim]
 * @param V_new       New value in FP32 [batch_heads, head_dim]
 * @param O           Output in FP32 [batch_heads, head_dim]
 * @param batch_heads Number of batch * heads
 * @param cache_pos   Current cache position (0-indexed)
 * @param head_dim    Head dimension
 * @return 0 on success, -1 on failure
 */
int flash_attention_int8_decode_fp32(
    const float* Q,
    const float* K_new,
    const float* V_new,
    float* O,
    int batch_heads,
    int cache_pos,
    int head_dim
);

/**
 * Get INT8 Flash Attention status
 *
 * @param initialized  Output: 1 if initialized
 * @param max_cache    Output: Maximum cache length
 * @param max_bh       Output: Maximum batch*heads
 * @param h_dim        Output: Head dimension
 */
void flash_attention_int8_info(int* initialized, int* max_cache, int* max_bh, int* h_dim);

#ifdef __cplusplus
}
#endif

#endif // FLASH_ATTENTION_INT8_H
