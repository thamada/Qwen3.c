#include "vk_alloc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern uint32_t vk_ctx_find_memory_type(VkPhysicalDevice phys, uint32_t type_bits,
                                        VkMemoryPropertyFlags props);

typedef struct {
    VkBuffer       buffer;
    VkDeviceMemory memory;
    size_t         size;
    int            in_use;
} DevBuf;

static VkCtx   *g_ctx;
static DevBuf  *g_bufs;
static int      g_nbufs;
static int      g_capbufs;
static size_t   g_allocated;

static int buf_index(const void *ptr)
{
    uintptr_t id = (uintptr_t)ptr;
    if (id == 0) return -1;
    return (int)(id - 1);
}

void vk_device_init(void)
{
    if (g_ctx) return;
    g_ctx = vk_ctx_create();
    if (!g_ctx) {
        fprintf(stderr, "Failed to create Vulkan context\n");
        exit(1);
    }
}

VkCtx *vk_device_ctx(void)
{
    vk_device_init();
    return g_ctx;
}

void vk_device_shutdown(void)
{
    if (!g_ctx) return;
    for (int i = 0; i < g_nbufs; i++) {
        if (!g_bufs[i].in_use) continue;
        vkDestroyBuffer(g_ctx->device, g_bufs[i].buffer, NULL);
        vkFreeMemory(g_ctx->device, g_bufs[i].memory, NULL);
    }
    free(g_bufs);
    g_bufs = NULL;
    g_nbufs = g_capbufs = 0;
    g_allocated = 0;
    vk_ctx_destroy(g_ctx);
    g_ctx = NULL;
}

static int alloc_slot(void)
{
    for (int i = 0; i < g_nbufs; i++)
        if (!g_bufs[i].in_use) return i;
    if (g_nbufs >= g_capbufs) {
        int nc = g_capbufs ? g_capbufs * 2 : 64;
        DevBuf *nb = (DevBuf *)realloc(g_bufs, (size_t)nc * sizeof(DevBuf));
        if (!nb) return -1;
        memset(nb + g_nbufs, 0, (size_t)(nc - g_nbufs) * sizeof(DevBuf));
        g_bufs = nb;
        g_capbufs = nc;
    }
    return g_nbufs++;
}

void *vk_malloc(size_t bytes)
{
    if (bytes == 0) bytes = 1;
    vk_device_init();

    int slot = alloc_slot();
    if (slot < 0) return NULL;

    VkDevice dev = g_ctx->device;
    VkPhysicalDevice phys = g_ctx->phys;

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkResult vr = vkCreateBuffer(dev, &bci, NULL, &g_bufs[slot].buffer);
    if (vr != VK_SUCCESS) return NULL;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, g_bufs[slot].buffer, &req);

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = vk_ctx_find_memory_type(
            phys, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
    };
    vr = vkAllocateMemory(dev, &mai, NULL, &g_bufs[slot].memory);
    if (vr != VK_SUCCESS) {
        vkDestroyBuffer(dev, g_bufs[slot].buffer, NULL);
        return NULL;
    }
    vkBindBufferMemory(dev, g_bufs[slot].buffer, g_bufs[slot].memory, 0);

    g_bufs[slot].size = bytes;
    g_bufs[slot].in_use = 1;
    g_allocated += bytes;
    return (void *)(uintptr_t)(slot + 1);
}

void vk_free(void *ptr)
{
    int idx = buf_index(ptr);
    if (idx < 0 || idx >= g_nbufs || !g_bufs[idx].in_use) return;
    VkDevice dev = g_ctx->device;
    g_allocated -= g_bufs[idx].size;
    vkDestroyBuffer(dev, g_bufs[idx].buffer, NULL);
    vkFreeMemory(dev, g_bufs[idx].memory, NULL);
    memset(&g_bufs[idx], 0, sizeof(g_bufs[idx]));
}

VkBuffer vk_ptr_buffer(const void *ptr)
{
    int idx = buf_index(ptr);
    if (idx < 0 || idx >= g_nbufs) return VK_NULL_HANDLE;
    return g_bufs[idx].buffer;
}

size_t vk_ptr_offset(const void *ptr)
{
    (void)ptr;
    return 0;
}

static void copy_buffer(VkBuffer src, VkBuffer dst, size_t size)
{
    VkCommandBuffer cmd = vk_ctx_begin_onetime(g_ctx);
    VkBufferCopy region = { 0, 0, size };
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    vk_ctx_submit_and_wait(g_ctx, cmd);
}

void vk_memcpy_h2d(void *dst, const void *src, size_t nbytes)
{
    vk_memcpy_h2d_offset(dst, 0, src, nbytes);
}

void vk_memcpy_h2d_offset(void *dst, size_t dst_off, const void *src, size_t nbytes)
{
    if (nbytes == 0) return;
    vk_device_init();
    int idx = buf_index(dst);
    if (idx < 0) return;

    VkDevice dev = g_ctx->device;
    VkPhysicalDevice phys = g_ctx->phys;

    VkBuffer staging;
    VkDeviceMemory staging_mem;
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = nbytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    vkCreateBuffer(dev, &bci, NULL, &staging);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, staging, &req);
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = vk_ctx_find_memory_type(
            phys, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
    };
    vkAllocateMemory(dev, &mai, NULL, &staging_mem);
    vkBindBufferMemory(dev, staging, staging_mem, 0);

    void *mapped;
    vkMapMemory(dev, staging_mem, 0, nbytes, 0, &mapped);
    memcpy(mapped, src, nbytes);
    vkUnmapMemory(dev, staging_mem);

    VkCommandBuffer cmd = vk_ctx_begin_onetime(g_ctx);
    VkBufferCopy region = { 0, dst_off, nbytes };
    vkCmdCopyBuffer(cmd, staging, g_bufs[idx].buffer, 1, &region);
    vk_ctx_submit_and_wait(g_ctx, cmd);

    vkDestroyBuffer(dev, staging, NULL);
    vkFreeMemory(dev, staging_mem, NULL);
}

void vk_memcpy_d2h(void *dst, const void *src, size_t nbytes)
{
    if (nbytes == 0) return;
    vk_device_init();
    int idx = buf_index(src);
    if (idx < 0) return;

    VkDevice dev = g_ctx->device;
    VkPhysicalDevice phys = g_ctx->phys;

    VkBuffer staging;
    VkDeviceMemory staging_mem;
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = nbytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    vkCreateBuffer(dev, &bci, NULL, &staging);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, staging, &req);
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = vk_ctx_find_memory_type(
            phys, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
    };
    vkAllocateMemory(dev, &mai, NULL, &staging_mem);
    vkBindBufferMemory(dev, staging, staging_mem, 0);

    copy_buffer(g_bufs[idx].buffer, staging, nbytes);

    void *mapped;
    vkMapMemory(dev, staging_mem, 0, nbytes, 0, &mapped);
    memcpy(dst, mapped, nbytes);
    vkUnmapMemory(dev, staging_mem);

    vkDestroyBuffer(dev, staging, NULL);
    vkFreeMemory(dev, staging_mem, NULL);
}

void vk_device_sync(void)
{
    if (!g_ctx) return;
    vkDeviceWaitIdle(g_ctx->device);
}
