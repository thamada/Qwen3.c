#ifndef VK_CONTEXT_H
#define VK_CONTEXT_H

#include <vulkan/vulkan.h>
#include <stddef.h>
#include <stdint.h>

typedef struct VkCtx {
    VkInstance       instance;
    VkPhysicalDevice phys;
    VkDevice         device;
    VkQueue          queue;
    uint32_t         queue_family;
    VkCommandPool    cmd_pool;
    VkDescriptorPool desc_pool;
    char             dev_name[256];
} VkCtx;

VkCtx *vk_ctx_create(void);
void   vk_ctx_destroy(VkCtx *ctx);

VkDevice              vk_ctx_device(const VkCtx *ctx);
VkPhysicalDevice      vk_ctx_phys(const VkCtx *ctx);
VkQueue               vk_ctx_queue(const VkCtx *ctx);
uint32_t              vk_ctx_queue_family(const VkCtx *ctx);
VkCommandPool         vk_ctx_cmd_pool(const VkCtx *ctx);
VkDescriptorPool      vk_ctx_desc_pool(const VkCtx *ctx);

void vk_ctx_submit_and_wait(VkCtx *ctx, VkCommandBuffer cmd);
VkCommandBuffer vk_ctx_begin_onetime(VkCtx *ctx);

void vk_ctx_get_memory_info(const VkCtx *ctx, size_t *total_bytes, size_t *used_bytes);
void vk_ctx_get_device_name(const VkCtx *ctx, char *buf, size_t cap);

#endif
