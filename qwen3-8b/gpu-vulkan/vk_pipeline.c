#include "vk_pipeline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    VkShaderModule         module;
    VkPipeline             pipeline;
    VkPipelineLayout       layout;
    VkDescriptorSetLayout  dsl;
    uint32_t               n_bindings;
} VkPipe;

static VkPipe g_pipes[VKP_COUNT];

static const char *pipe_names[VKP_COUNT] = {
    "emb_f16", "emb_f16_batch",
    "mm_f16_gemv", "mm_f16_gemv_batch",
    "rmsnorm", "rmsnorm_batch",
    "rmsnorm_head", "rmsnorm_head_batch",
    "rope", "rope_prefill_batch",
    "vec_add", "vec_add_batch",
    "silu_mul", "silu_mul_batch",
    "kv_write", "kv_write_batch",
    "flash_attn_decode", "flash_attn_prefill",
};

static uint32_t binding_count(VkPipeKind k)
{
    switch (k) {
    case VKP_EMB_F16: return 2;
    case VKP_EMB_F16_BATCH: return 3;
    case VKP_MM_F16_GEMV:
    case VKP_MM_F16_GEMV_BATCH: return 3;
    case VKP_RMSNORM:
    case VKP_RMSNORM_BATCH: return 3;
    case VKP_RMSNORM_HEAD:
    case VKP_RMSNORM_HEAD_BATCH: return 2;
    case VKP_ROPE:
    case VKP_ROPE_PREFILL_BATCH: return 1;
    case VKP_VEC_ADD:
    case VKP_VEC_ADD_BATCH: return 2;
    case VKP_SILU_MUL:
    case VKP_SILU_MUL_BATCH: return 2;
    case VKP_KV_WRITE:
    case VKP_KV_WRITE_BATCH: return 2;
    case VKP_FLASH_ATTN_DECODE:
    case VKP_FLASH_ATTN_PREFILL: return 4;
    default: return 0;
    }
}

static void vk_die(const char *msg, VkResult res)
{
    fprintf(stderr, "Vulkan pipeline error: %s (VkResult=%d)\n", msg, (int)res);
    exit(1);
}

static uint32_t *load_spv(const char *path, size_t *out_words)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open shader: %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || (sz % 4) != 0) {
        fclose(f);
        return NULL;
    }
    uint32_t *code = (uint32_t *)malloc((size_t)sz);
    if (fread(code, 1, (size_t)sz, f) != (size_t)sz) {
        free(code);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_words = (size_t)sz / 4;
    return code;
}

static void create_pipe(VkCtx *ctx, VkPipeKind kind, const char *spv_path)
{
    VkDevice dev = vk_ctx_device(ctx);
    size_t nwords = 0;
    uint32_t *code = load_spv(spv_path, &nwords);
    if (!code) {
        fprintf(stderr, "Failed to load SPIR-V: %s\n", spv_path);
        exit(1);
    }

    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = nwords * 4,
        .pCode = code,
    };
    VkShaderModule mod;
    VkResult vr = vkCreateShaderModule(dev, &smci, NULL, &mod);
    free(code);
    if (vr != VK_SUCCESS) vk_die("vkCreateShaderModule", vr);

    uint32_t nb = binding_count(kind);
    VkDescriptorSetLayoutBinding binds[4];
    for (uint32_t i = 0; i < nb; i++) {
        binds[i] = (VkDescriptorSetLayoutBinding){
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }

    VkDescriptorSetLayout dsl;
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = nb,
        .pBindings = binds,
    };
    vr = vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl);
    if (vr != VK_SUCCESS) vk_die("vkCreateDescriptorSetLayout", vr);

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 128,
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr,
    };
    VkPipelineLayout layout;
    vr = vkCreatePipelineLayout(dev, &plci, NULL, &layout);
    if (vr != VK_SUCCESS) vk_die("vkCreatePipelineLayout", vr);

    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = mod,
            .pName = "main",
        },
        .layout = layout,
    };
    VkPipeline pipeline;
    vr = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipeline);
    if (vr != VK_SUCCESS) vk_die("vkCreateComputePipelines", vr);

    g_pipes[kind].module = mod;
    g_pipes[kind].pipeline = pipeline;
    g_pipes[kind].layout = layout;
    g_pipes[kind].dsl = dsl;
    g_pipes[kind].n_bindings = nb;
}

void vk_pipelines_init(VkCtx *ctx, const char *shader_dir)
{
    char path[512];
    for (int i = 0; i < VKP_COUNT; i++) {
        snprintf(path, sizeof path, "%s/%s.spv", shader_dir, pipe_names[i]);
        create_pipe(ctx, (VkPipeKind)i, path);
    }
}

void vk_pipelines_shutdown(VkCtx *ctx)
{
    VkDevice dev = vk_ctx_device(ctx);
    for (int i = 0; i < VKP_COUNT; i++) {
        if (g_pipes[i].pipeline) vkDestroyPipeline(dev, g_pipes[i].pipeline, NULL);
        if (g_pipes[i].layout) vkDestroyPipelineLayout(dev, g_pipes[i].layout, NULL);
        if (g_pipes[i].dsl) vkDestroyDescriptorSetLayout(dev, g_pipes[i].dsl, NULL);
        if (g_pipes[i].module) vkDestroyShaderModule(dev, g_pipes[i].module, NULL);
        memset(&g_pipes[i], 0, sizeof(g_pipes[i]));
    }
}

void vk_dispatch(VkCtx *ctx, VkPipeKind kind,
                 VkBuffer bufs[], uint32_t n_bufs,
                 const void *push, uint32_t push_size,
                 uint32_t gx, uint32_t gy, uint32_t gz)
{
    VkDevice dev = vk_ctx_device(ctx);
    VkPipe *p = &g_pipes[kind];
    if (n_bufs != p->n_bindings) {
        fprintf(stderr, "vk_dispatch: binding mismatch for %s\n", pipe_names[kind]);
        exit(1);
    }

    for (uint32_t i = 0; i < n_bufs; i++) {
        if (bufs[i] == VK_NULL_HANDLE) {
            fprintf(stderr, "vk_dispatch: null buffer binding %u for %s\n", i, pipe_names[kind]);
            exit(1);
        }
    }

    VkDescriptorSet set;
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk_ctx_desc_pool(ctx),
        .descriptorSetCount = 1,
        .pSetLayouts = &p->dsl,
    };
    VkResult vr = vkAllocateDescriptorSets(dev, &dsai, &set);
    if (vr != VK_SUCCESS) vk_die("vkAllocateDescriptorSets", vr);

    VkWriteDescriptorSet writes[4];
    VkDescriptorBufferInfo dbi[4];
    for (uint32_t i = 0; i < n_bufs; i++) {
        dbi[i] = (VkDescriptorBufferInfo){ bufs[i], 0, VK_WHOLE_SIZE };
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &dbi[i],
        };
    }
    vkUpdateDescriptorSets(dev, n_bufs, writes, 0, NULL);

    VkCommandBuffer cmd = vk_ctx_begin_onetime(ctx);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout,
                            0, 1, &set, 0, NULL);
    if (push && push_size > 0)
        vkCmdPushConstants(cmd, p->layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, push_size, push);
    vkCmdDispatch(cmd, gx, gy, gz);
    vk_ctx_submit_and_wait(ctx, cmd);
    vkFreeDescriptorSets(dev, vk_ctx_desc_pool(ctx), 1, &set);
}
