#ifndef VK_PIPELINE_H
#define VK_PIPELINE_H

#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#include "vk_context.h"

typedef enum {
    VKP_EMB_F16,
    VKP_EMB_F16_BATCH,
    VKP_MM_F16_GEMV,
    VKP_MM_F16_GEMV_BATCH,
    VKP_RMSNORM,
    VKP_RMSNORM_BATCH,
    VKP_RMSNORM_HEAD,
    VKP_RMSNORM_HEAD_BATCH,
    VKP_ROPE,
    VKP_ROPE_PREFILL_BATCH,
    VKP_VEC_ADD,
    VKP_VEC_ADD_BATCH,
    VKP_SILU_MUL,
    VKP_SILU_MUL_BATCH,
    VKP_KV_WRITE,
    VKP_KV_WRITE_BATCH,
    VKP_FLASH_ATTN_DECODE,
    VKP_FLASH_ATTN_PREFILL,
    VKP_COUNT
} VkPipeKind;

void vk_pipelines_init(VkCtx *ctx, const char *shader_dir);
void vk_pipelines_shutdown(VkCtx *ctx);

void vk_dispatch(VkCtx *ctx, VkPipeKind kind,
                 VkBuffer bufs[], uint32_t n_bufs,
                 const void *push, uint32_t push_size,
                 uint32_t gx, uint32_t gy, uint32_t gz);

#endif
