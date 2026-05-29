#include "vk_context.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void vk_die(const char *msg, VkResult res)
{
    fprintf(stderr, "Vulkan error: %s (VkResult=%d)\n", msg, (int)res);
    exit(1);
}

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_bits,
                                 VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    vk_die("find_memory_type", VK_ERROR_FEATURE_NOT_PRESENT);
    return 0;
}

VkCtx *vk_ctx_create(void)
{
    VkCtx *ctx = (VkCtx *)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "qwen3-vulkan",
        .applicationVersion = 1,
        .pEngineName = "qwen3",
        .engineVersion = 1,
        .apiVersion = VK_API_VERSION_1_1,
    };

    const char *exts[] = { VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames = exts,
        .flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
    };

    VkResult vr = vkCreateInstance(&ici, NULL, &ctx->instance);
    if (vr != VK_SUCCESS) vk_die("vkCreateInstance", vr);

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &ndev, NULL);
    if (ndev == 0) vk_die("no Vulkan devices", VK_ERROR_INITIALIZATION_FAILED);

    VkPhysicalDevice *devs = (VkPhysicalDevice *)malloc(ndev * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx->instance, &ndev, devs);

    ctx->phys = devs[0];
    for (uint32_t i = 0; i < ndev; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devs[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            ctx->phys = devs[i];
            break;
        }
    }
    free(devs);

    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(ctx->phys, &props);
        snprintf(ctx->dev_name, sizeof ctx->dev_name, "%s", props.deviceName);
    }

    float qprio = 1.f;
    uint32_t ndev_ext = 0;
    const char **dev_exts = NULL;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0,
        .queueCount = 1,
        .pQueuePriorities = &qprio,
    };

    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys, &nqf, NULL);
    VkQueueFamilyProperties *qfp =
        (VkQueueFamilyProperties *)calloc(nqf, sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys, &nqf, qfp);
    for (uint32_t i = 0; i < nqf; i++) {
        if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            qci.queueFamilyIndex = i;
            ctx->queue_family = i;
            break;
        }
    }
    free(qfp);

    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = ndev_ext,
        .ppEnabledExtensionNames = ndev_ext ? dev_exts : NULL,
    };
    vr = vkCreateDevice(ctx->phys, &dci, NULL, &ctx->device);
    if (vr != VK_SUCCESS) vk_die("vkCreateDevice", vr);

    vkGetDeviceQueue(ctx->device, ctx->queue_family, 0, &ctx->queue);

    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx->queue_family,
    };
    vr = vkCreateCommandPool(ctx->device, &cpci, NULL, &ctx->cmd_pool);
    if (vr != VK_SUCCESS) vk_die("vkCreateCommandPool", vr);

    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096 };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 4096,
        .poolSizeCount = 1,
        .pPoolSizes = &dps,
    };
    vr = vkCreateDescriptorPool(ctx->device, &dpci, NULL, &ctx->desc_pool);
    if (vr != VK_SUCCESS) vk_die("vkCreateDescriptorPool", vr);

    return ctx;
}

void vk_ctx_destroy(VkCtx *ctx)
{
    if (!ctx) return;
    if (ctx->desc_pool) vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
    if (ctx->cmd_pool) vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
    if (ctx->device) vkDestroyDevice(ctx->device, NULL);
    if (ctx->instance) vkDestroyInstance(ctx->instance, NULL);
    free(ctx);
}

VkDevice         vk_ctx_device(const VkCtx *ctx)       { return ctx->device; }
VkPhysicalDevice vk_ctx_phys(const VkCtx *ctx)         { return ctx->phys; }
VkQueue          vk_ctx_queue(const VkCtx *ctx)        { return ctx->queue; }
uint32_t         vk_ctx_queue_family(const VkCtx *ctx) { return ctx->queue_family; }
VkCommandPool    vk_ctx_cmd_pool(const VkCtx *ctx)     { return ctx->cmd_pool; }
VkDescriptorPool vk_ctx_desc_pool(const VkCtx *ctx)    { return ctx->desc_pool; }

void vk_ctx_get_device_name(const VkCtx *ctx, char *buf, size_t cap)
{
    if (!buf || cap == 0) return;
    snprintf(buf, cap, "%s", ctx->dev_name);
}

void vk_ctx_get_memory_info(const VkCtx *ctx, size_t *total_bytes, size_t *used_bytes)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(ctx->phys, &mp);
    size_t total = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            total += mp.memoryHeaps[i].size;
    if (total_bytes) *total_bytes = total;
    if (used_bytes) *used_bytes = 0;
}

VkCommandBuffer vk_ctx_begin_onetime(VkCtx *ctx)
{
    VkCommandBufferAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd;
    VkResult vr = vkAllocateCommandBuffers(ctx->device, &ai, &cmd);
    if (vr != VK_SUCCESS) vk_die("vkAllocateCommandBuffers", vr);

    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vr = vkBeginCommandBuffer(cmd, &bi);
    if (vr != VK_SUCCESS) vk_die("vkBeginCommandBuffer", vr);
    return cmd;
}

void vk_ctx_submit_and_wait(VkCtx *ctx, VkCommandBuffer cmd)
{
    VkResult vr = vkEndCommandBuffer(cmd);
    if (vr != VK_SUCCESS) vk_die("vkEndCommandBuffer", vr);

    VkFence fence;
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    vr = vkCreateFence(ctx->device, &fci, NULL, &fence);
    if (vr != VK_SUCCESS) vk_die("vkCreateFence", vr);

    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    vr = vkQueueSubmit(ctx->queue, 1, &si, fence);
    if (vr != VK_SUCCESS) vk_die("vkQueueSubmit", vr);

    vr = vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (vr != VK_SUCCESS) vk_die("vkWaitForFences", vr);

    vkDestroyFence(ctx->device, fence, NULL);
    vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cmd);
}

/* exported for vk_alloc.c */
uint32_t vk_ctx_find_memory_type(VkPhysicalDevice phys, uint32_t type_bits,
                                 VkMemoryPropertyFlags props)
{
    return find_memory_type(phys, type_bits, props);
}
