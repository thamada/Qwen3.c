#ifndef VK_ALLOC_H
#define VK_ALLOC_H

#include <stddef.h>
#include <vulkan/vulkan.h>

#include "vk_context.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VKCHK(x) do { \
    if (!(x)) { \
        fprintf(stderr, "%s:%d VK alloc error\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

void vk_device_init(void);
void vk_device_shutdown(void);
VkCtx *vk_device_ctx(void);

void *vk_malloc(size_t bytes);
void  vk_free(void *ptr);

void vk_memcpy_h2d(void *dst, const void *src, size_t nbytes);
void vk_memcpy_h2d_offset(void *dst, size_t dst_off, const void *src, size_t nbytes);
void vk_memcpy_d2h(void *dst, const void *src, size_t nbytes);
void vk_device_sync(void);

VkBuffer vk_ptr_buffer(const void *ptr);
size_t   vk_ptr_offset(const void *ptr);

#ifdef __cplusplus
}
#endif

#endif
