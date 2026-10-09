/*
Copyright (C) 2026 Aram Fingal

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
*/

// vk_metal_pipeline.cpp -- descriptor sets, shader modules and pipelines.
//
// A shader module is the MSL source that mslgen wrote, with a header that says which
// argument index each Vulkan (set, binding) got. Metal lays the members of an argument
// buffer out in declaration order, and a shader only declares what it uses, so where a
// binding ends up differs between shaders. When a pipeline is created its reflection
// gives the offset of every argument index; from that each stage gets an ArgLayout per
// descriptor set. A descriptor set keeps its contents on the CPU and encodes them into
// a Metal buffer per ArgLayout the first time it is bound with a pipeline using it, and
// again after it was updated.

#include "vk_metal_internal.hpp"

#include <set>
#include <stdlib.h>
#include <string.h>

namespace vkmtl {

const VkDescriptorSetLayoutBinding *SetLayoutVk::find(uint32_t binding) const
{
    for (const auto &b : bindings) {
        if (b.binding == binding)
            return &b;
    }
    return nullptr;
}

const DescSet::Encoded &DescSet::encode(const ArgLayout *arg_layout)
{
    Encoded &enc = encoded[arg_layout];
    if (enc.buffer && enc.version == version)
        return enc;

    size_t size = arg_layout->fixed_size;
    for (const auto &e : arg_layout->entries) {
        if (e.count)
            continue;
        const VkDescriptorSetLayoutBinding *b = layout->find(e.binding);
        size_t end = (size_t)e.offset + 8 * (b ? b->descriptorCount : 1);
        if (end > size)
            size = end;
    }
    if (size < 8)
        size = 8;

    // A new buffer every time: a frame in flight may still be reading the old one.
    if (enc.buffer)
        enc.buffer->release();
    enc.buffer = g.device->newBuffer(size, MTL::ResourceStorageModeShared);
    enc.version = version;
    enc.written.clear();
    enc.read.clear();
    std::set<const MTL::Resource *> seen;

    uint64_t *out = static_cast<uint64_t *>(enc.buffer->contents());
    memset(out, 0, size);

    for (const auto &e : arg_layout->entries) {
        auto it = slots.find(e.binding);
        if (it == slots.end())
            continue;
        const std::vector<Slot> &list = it->second;
        size_t count = e.count && e.count < list.size() ? e.count : list.size();

        for (size_t i = 0; i < count; i++) {
            if (arg_layout->samplers_only) {
                out[e.offset / 8 + i] = list[i].sampler_value;
                continue;
            }
            const Slot &slot = list[i];
            if (e.offset >= 0)
                out[e.offset / 8 + i] = slot.value;

            if (!slot.resource || !seen.insert(slot.resource).second)
                continue;
            if (slot.type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                enc.read.push_back(slot.resource);
            else if (slot.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || slot.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                enc.written.push_back(slot.resource);
            if (e.sampler_offset >= 0)
                out[e.sampler_offset / 8 + i] = slot.sampler_value;
        }
    }
    return enc;
}

// ------------------------------------------------------------------------------ shaders

static ShaderModule *parse_module(const char *text, size_t size)
{
    static const char magic[] = "/*VKMTL\n";
    if (size < sizeof(magic) || strncmp(text, magic, sizeof(magic) - 1)) {
        vkmtl_error("shader module is not a translated Metal shader\n");
        return nullptr;
    }

    ShaderModule *module = new ShaderModule();
    module->source.assign(text, size);
    module->local_size[0] = module->local_size[1] = module->local_size[2] = 1;

    size_t end = module->source.find("*/");
    std::string header = module->source.substr(sizeof(magic) - 1, end - (sizeof(magic) - 1));

    size_t pos = 0;
    while (pos < header.size()) {
        size_t eol = header.find('\n', pos);
        if (eol == std::string::npos)
            eol = header.size();
        std::string line = header.substr(pos, eol - pos);
        pos = eol + 1;

        char word[64];
        ShaderModule::Bind b = {};
        int runtime = 0;
        if (sscanf(line.c_str(), "name %63s", word) == 1) {
            module->name = word;
        } else if (sscanf(line.c_str(), "stage %63s", word) == 1) {
            module->stage = word;
        } else if (sscanf(line.c_str(), "entry %63s", word) == 1) {
            module->entry = word;
        } else if (!strncmp(line.c_str(), "push ", 5)) {
            module->has_push = true;
        } else if (sscanf(line.c_str(), "local %u %u %u", &module->local_size[0], &module->local_size[1],
                          &module->local_size[2]) == 3) {
        } else if (sscanf(line.c_str(), "bind %u %u %u %d %d %d", &b.set, &b.binding, &b.count, &b.id,
                          &b.sampler_id, &runtime) == 6) {
            b.runtime = runtime != 0;
            module->binds.push_back(b);
        }
    }
    return module;
}

MTL::Function *ShaderModule::function(const VkSpecializationInfo *spec)
{
    NS::Error *error = nullptr;

    if (!library) {
        MTL::CompileOptions *options = MTL::CompileOptions::alloc()->init();
        options->setLanguageVersion(MTL::LanguageVersion3_1);
        // The shaders test for NaN and infinity, which fast math would optimize away.
        options->setMathMode(MTL::MathModeSafe);
        library = g.device->newLibrary(nsstr(source.c_str()), options, &error);
        options->release();
        if (!library) {
            vkmtl_error("compiling a shader failed: %s\n",
                        error ? error->localizedDescription()->utf8String() : "unknown error");
            return nullptr;
        }
    }

    // SPIRV-Cross turns specialization constants into Metal function constants with
    // the constant ID as index; all of vkpt's are 32-bit integers.
    MTL::FunctionConstantValues *values = MTL::FunctionConstantValues::alloc()->init();
    if (spec) {
        for (uint32_t i = 0; i < spec->mapEntryCount; i++) {
            const VkSpecializationMapEntry &e = spec->pMapEntries[i];
            uint32_t value = 0;
            memcpy(&value, (const char *)spec->pData + e.offset, e.size < 4 ? e.size : 4);
            values->setConstantValue(&value, MTL::DataTypeUInt, (NS::UInteger)e.constantID);
        }
    }
    MTL::Function *function = library->newFunction(nsstr(entry.c_str()), values, &error);
    values->release();
    if (!function)
        vkmtl_error("specializing a shader failed: %s\n",
                    error ? error->localizedDescription()->utf8String() : "unknown error");
    return function;
}

static const ArgLayout *intern_layout(const ArgLayout &layout)
{
    for (const ArgLayout *l : g.arg_layouts) {
        if (l->fixed_size != layout.fixed_size || l->samplers_only != layout.samplers_only
            || l->entries.size() != layout.entries.size())
            continue;
        if (!memcmp(l->entries.data(), layout.entries.data(), layout.entries.size() * sizeof(ArgLayout::Entry)))
            return l;
    }
    g.arg_layouts.push_back(new ArgLayout(layout));
    return g.arg_layouts.back();
}

void Stage::reflect(NS::Array *bindings)
{
    for (NS::UInteger i = 0; i < bindings->count(); i++) {
        MTL::Binding *binding = bindings->object<MTL::Binding>(i);
        if (binding->type() != MTL::BindingTypeBuffer)
            continue;

        NS::UInteger index = binding->index();
        if (index == PUSH_CONSTANT_BUFFER_INDEX) {
            has_push = true;
            continue;
        }
        if (index >= MAX_SETS)
            continue;

        MTL::StructType *type = static_cast<MTL::BufferBinding *>(binding)->bufferStructType();
        if (!type)
            continue;

        std::map<int, int> offsets;  // argument index -> offset
        NS::Array *members = type->members();
        for (NS::UInteger m = 0; m < members->count(); m++) {
            MTL::StructMember *member = members->object<MTL::StructMember>(m);
            offsets[(int)member->argumentIndex()] = (int)member->offset();
        }

        ArgLayout layout;
        layout.fixed_size = 0;
        layout.samplers_only = index >= SAMPLER_SET_OFFSET;
        for (const ShaderModule::Bind &b : module->binds) {
            if (b.set != index)
                continue;
            auto offset = offsets.find(b.id);
            auto sampler = offsets.find(b.sampler_id);

            ArgLayout::Entry e = {};
            e.binding = b.binding;
            e.count = b.runtime ? 0 : b.count;
            e.offset = offset != offsets.end() ? offset->second : -1;
            e.sampler_offset = sampler != offsets.end() ? sampler->second : -1;
            if (e.offset < 0 && e.sampler_offset < 0)
                continue;
            layout.entries.push_back(e);

            size_t n = b.runtime ? 1 : b.count;
            if (e.offset >= 0 && e.offset + 8 * n > layout.fixed_size)
                layout.fixed_size = e.offset + 8 * n;
            if (e.sampler_offset >= 0 && e.sampler_offset + 8 * n > layout.fixed_size)
                layout.fixed_size = e.sampler_offset + 8 * n;
        }
        sets[index] = intern_layout(layout);
    }
}

}  // namespace vkmtl

using namespace vkmtl;

namespace {

struct DescPool {
    std::vector<DescSet *> sets;
};

void free_set(DescSet *set)
{
    for (auto &b : set->slots) {
        for (Slot &slot : b.second) {
            if (slot.resource)
                slot.resource->release();
        }
    }
    for (auto &e : set->encoded) {
        if (e.second.buffer)
            e.second.buffer->release();
    }
    delete set;
}

MTL::VertexFormat vertex_format(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_R32_SFLOAT:          return MTL::VertexFormatFloat;
    case VK_FORMAT_R32G32_SFLOAT:       return MTL::VertexFormatFloat2;
    case VK_FORMAT_R32G32B32_SFLOAT:    return MTL::VertexFormatFloat3;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return MTL::VertexFormatFloat4;
    case VK_FORMAT_R8G8B8A8_UNORM:      return MTL::VertexFormatUChar4Normalized;
    case VK_FORMAT_R32_UINT:            return MTL::VertexFormatUInt;
    default:
        vkmtl_error("unsupported vertex format %d\n", (int)format);
        return MTL::VertexFormatInvalid;
    }
}

MTL::BlendFactor blend_factor(VkBlendFactor factor)
{
    switch (factor) {
    case VK_BLEND_FACTOR_ZERO:                return MTL::BlendFactorZero;
    case VK_BLEND_FACTOR_SRC_ALPHA:           return MTL::BlendFactorSourceAlpha;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: return MTL::BlendFactorOneMinusSourceAlpha;
    case VK_BLEND_FACTOR_DST_ALPHA:           return MTL::BlendFactorDestinationAlpha;
    case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: return MTL::BlendFactorOneMinusDestinationAlpha;
    case VK_BLEND_FACTOR_SRC_COLOR:           return MTL::BlendFactorSourceColor;
    case VK_BLEND_FACTOR_DST_COLOR:           return MTL::BlendFactorDestinationColor;
    default:                                  return MTL::BlendFactorOne;
    }
}

MTL::CompareFunction compare_function(VkCompareOp op)
{
    switch (op) {
    case VK_COMPARE_OP_NEVER:            return MTL::CompareFunctionNever;
    case VK_COMPARE_OP_LESS:             return MTL::CompareFunctionLess;
    case VK_COMPARE_OP_EQUAL:            return MTL::CompareFunctionEqual;
    case VK_COMPARE_OP_LESS_OR_EQUAL:    return MTL::CompareFunctionLessEqual;
    case VK_COMPARE_OP_GREATER:          return MTL::CompareFunctionGreater;
    case VK_COMPARE_OP_NOT_EQUAL:        return MTL::CompareFunctionNotEqual;
    case VK_COMPARE_OP_GREATER_OR_EQUAL: return MTL::CompareFunctionGreaterEqual;
    default:                             return MTL::CompareFunctionAlways;
    }
}

const char *error_text(NS::Error *error)
{
    return error && error->localizedDescription() ? error->localizedDescription()->utf8String() : "unknown error";
}

}  // namespace

extern "C" {

// ------------------------------------------------------------------------------ descriptor sets

VkResult vkCreateDescriptorSetLayout(VkDevice, const VkDescriptorSetLayoutCreateInfo *info,
                                     const VkAllocationCallbacks *, VkDescriptorSetLayout *out)
{
    SetLayoutVk *layout = new SetLayoutVk();
    layout->bindings.assign(info->pBindings, info->pBindings + info->bindingCount);
    for (auto &b : layout->bindings)
        b.pImmutableSamplers = nullptr;
    *out = reinterpret_cast<VkDescriptorSetLayout>(layout);
    return VK_SUCCESS;
}

void vkDestroyDescriptorSetLayout(VkDevice, VkDescriptorSetLayout handle, const VkAllocationCallbacks *)
{
    // Descriptor sets keep pointing at their layout, and vkpt destroys the two in
    // either order; layouts are few and small, so they are simply kept.
    (void)handle;
}

VkResult vkCreateDescriptorPool(VkDevice, const VkDescriptorPoolCreateInfo *, const VkAllocationCallbacks *,
                                VkDescriptorPool *out)
{
    *out = reinterpret_cast<VkDescriptorPool>(new DescPool());
    return VK_SUCCESS;
}

void vkDestroyDescriptorPool(VkDevice, VkDescriptorPool handle, const VkAllocationCallbacks *)
{
    DescPool *pool = reinterpret_cast<DescPool *>(handle);
    if (!pool)
        return;
    for (DescSet *set : pool->sets)
        free_set(set);
    delete pool;
}

VkResult vkAllocateDescriptorSets(VkDevice, const VkDescriptorSetAllocateInfo *info, VkDescriptorSet *out)
{
    DescPool *pool = reinterpret_cast<DescPool *>(info->descriptorPool);
    for (uint32_t i = 0; i < info->descriptorSetCount; i++) {
        DescSet *set = new DescSet();
        set->layout = VKMTL_HANDLE(SetLayoutVk, info->pSetLayouts[i]);
        set->version = 1;
        pool->sets.push_back(set);
        out[i] = reinterpret_cast<VkDescriptorSet>(set);
    }
    return VK_SUCCESS;
}

void vkUpdateDescriptorSets(VkDevice, uint32_t write_count, const VkWriteDescriptorSet *writes, uint32_t copy_count,
                            const VkCopyDescriptorSet *)
{
    if (copy_count)
        vkmtl_unsupported("copying descriptors");

    for (uint32_t w = 0; w < write_count; w++) {
        const VkWriteDescriptorSet &write = writes[w];
        DescSet *set = VKMTL_HANDLE(DescSet, write.dstSet);
        std::vector<Slot> &slots = set->slots[write.dstBinding];
        if (slots.size() < write.dstArrayElement + write.descriptorCount)
            slots.resize(write.dstArrayElement + write.descriptorCount);

        const VkWriteDescriptorSetAccelerationStructureKHR *accels = nullptr;
        for (const VkBaseInStructure *s = (const VkBaseInStructure *)write.pNext; s; s = s->pNext) {
            if (s->sType == VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR)
                accels = (const VkWriteDescriptorSetAccelerationStructureKHR *)s;
        }

        for (uint32_t i = 0; i < write.descriptorCount; i++) {
            Slot slot = {};
            slot.type = write.descriptorType;
            switch (write.descriptorType) {
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            case VK_DESCRIPTOR_TYPE_SAMPLER:
                if (write.pImageInfo[i].imageView) {
                    MTL::Texture *texture = VKMTL_HANDLE(ImageView, write.pImageInfo[i].imageView)->texture;
                    slot.resource = texture;
                    slot.value = texture->gpuResourceID()._impl;
                }
                if (write.pImageInfo[i].sampler)
                    slot.sampler_value = VKMTL_HANDLE(Sampler, write.pImageInfo[i].sampler)->state->gpuResourceID()._impl;
                break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                if (write.pBufferInfo[i].buffer) {
                    Buffer *buffer = VKMTL_HANDLE(Buffer, write.pBufferInfo[i].buffer);
                    slot.resource = buffer->mtl();
                    slot.value = buffer->gpu_address() + write.pBufferInfo[i].offset;
                }
                break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                if (write.pTexelBufferView[i]) {
                    MTL::Texture *texture = VKMTL_HANDLE(BufferView, write.pTexelBufferView[i])->texture;
                    slot.resource = texture;
                    slot.value = texture->gpuResourceID()._impl;
                }
                break;
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
                if (accels && accels->pAccelerationStructures[i]) {
                    MTL::AccelerationStructure *as = VKMTL_HANDLE(Accel, accels->pAccelerationStructures[i])->as;
                    slot.resource = as;
                    slot.value = as->gpuResourceID()._impl;
                }
                break;
            default:
                vkmtl_unsupported("descriptor type");
                break;
            }
            if (slot.resource)
                slot.resource->retain();
            Slot &old = slots[write.dstArrayElement + i];
            if (old.resource)
                old.resource->release();
            old = slot;
        }
        set->version++;
    }
}

// ------------------------------------------------------------------------------ shader modules

VkResult vkCreateShaderModule(VkDevice, const VkShaderModuleCreateInfo *info, const VkAllocationCallbacks *,
                              VkShaderModule *out)
{
    ShaderModule *module = parse_module((const char *)info->pCode, info->codeSize);
    if (!module)
        return VK_ERROR_INVALID_SHADER_NV;
    *out = reinterpret_cast<VkShaderModule>(module);
    return VK_SUCCESS;
}

void vkDestroyShaderModule(VkDevice, VkShaderModule handle, const VkAllocationCallbacks *)
{
    ShaderModule *module = VKMTL_HANDLE(ShaderModule, handle);
    if (!module)
        return;
    if (module->library)
        module->library->release();
    delete module;
}

// ------------------------------------------------------------------------------ pipelines

VkResult vkCreatePipelineLayout(VkDevice, const VkPipelineLayoutCreateInfo *, const VkAllocationCallbacks *,
                                VkPipelineLayout *out)
{
    static int dummy;
    *out = reinterpret_cast<VkPipelineLayout>(&dummy);
    return VK_SUCCESS;
}

void vkDestroyPipelineLayout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks *)
{
}

VkResult vkCreateComputePipelines(VkDevice, VkPipelineCache, uint32_t count, const VkComputePipelineCreateInfo *infos,
                                  const VkAllocationCallbacks *, VkPipeline *out)
{
    VkResult result = VK_SUCCESS;

    for (uint32_t i = 0; i < count; i++) {
        out[i] = VK_NULL_HANDLE;

        ShaderModule *module = VKMTL_HANDLE(ShaderModule, infos[i].stage.module);
        MTL::Function *function = module ? module->function(infos[i].stage.pSpecializationInfo) : nullptr;
        if (!function) {
            result = VK_ERROR_INITIALIZATION_FAILED;
            continue;
        }

        NS::Error *error = nullptr;
        MTL::ComputePipelineReflection *reflection = nullptr;
        MTL::ComputePipelineState *state = g.device->newComputePipelineState(
            function, MTL::PipelineOptionBindingInfo | MTL::PipelineOptionBufferTypeInfo, &reflection, &error);
        function->release();
        if (!state) {
            vkmtl_error("creating a compute pipeline failed: %s\n", error_text(error));
            result = VK_ERROR_INITIALIZATION_FAILED;
            continue;
        }

        Pipeline *pipeline = new Pipeline();
        pipeline->bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
        pipeline->compute = state;
        pipeline->stages[0].module = module;
        pipeline->stages[0].reflect(reflection->bindings());
        pipeline->threads_per_group = MTL::Size::Make(module->local_size[0], module->local_size[1], module->local_size[2]);
        out[i] = reinterpret_cast<VkPipeline>(pipeline);
    }
    return result;
}

VkResult vkCreateGraphicsPipelines(VkDevice, VkPipelineCache, uint32_t count, const VkGraphicsPipelineCreateInfo *infos,
                                   const VkAllocationCallbacks *, VkPipeline *out)
{
    VkResult result = VK_SUCCESS;

    for (uint32_t i = 0; i < count; i++) {
        const VkGraphicsPipelineCreateInfo &info = infos[i];
        out[i] = VK_NULL_HANDLE;

        Pipeline *pipeline = new Pipeline();
        pipeline->bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;

        MTL::RenderPipelineDescriptor *desc = MTL::RenderPipelineDescriptor::alloc()->init();
        bool ok = true;

        for (uint32_t s = 0; s < info.stageCount; s++) {
            const VkPipelineShaderStageCreateInfo &stage = info.pStages[s];
            ShaderModule *module = VKMTL_HANDLE(ShaderModule, stage.module);
            MTL::Function *function = module ? module->function(stage.pSpecializationInfo) : nullptr;
            if (!function) {
                ok = false;
                continue;
            }
            if (stage.stage == VK_SHADER_STAGE_VERTEX_BIT) {
                desc->setVertexFunction(function);
                pipeline->stages[0].module = module;
            } else {
                desc->setFragmentFunction(function);
                pipeline->stages[1].module = module;
            }
            function->release();
        }

        if (const VkPipelineVertexInputStateCreateInfo *vi = info.pVertexInputState) {
            MTL::VertexDescriptor *vertex = MTL::VertexDescriptor::vertexDescriptor();
            for (uint32_t a = 0; a < vi->vertexAttributeDescriptionCount; a++) {
                const VkVertexInputAttributeDescription &attr = vi->pVertexAttributeDescriptions[a];
                MTL::VertexAttributeDescriptor *ad = vertex->attributes()->object(attr.location);
                ad->setFormat(vertex_format(attr.format));
                ad->setOffset(attr.offset);
                ad->setBufferIndex(VERTEX_BUFFER_BASE_INDEX + attr.binding);
            }
            for (uint32_t b = 0; b < vi->vertexBindingDescriptionCount; b++) {
                const VkVertexInputBindingDescription &binding = vi->pVertexBindingDescriptions[b];
                MTL::VertexBufferLayoutDescriptor *ld = vertex->layouts()->object(VERTEX_BUFFER_BASE_INDEX + binding.binding);
                ld->setStride(binding.stride);
                ld->setStepFunction(binding.inputRate == VK_VERTEX_INPUT_RATE_INSTANCE
                                    ? MTL::VertexStepFunctionPerInstance : MTL::VertexStepFunctionPerVertex);
            }
            if (vi->vertexAttributeDescriptionCount)
                desc->setVertexDescriptor(vertex);
        }

        RenderPass *pass = VKMTL_HANDLE(RenderPass, info.renderPass);
        for (size_t c = 0; c < pass->color.size(); c++) {
            MTL::RenderPipelineColorAttachmentDescriptor *color = desc->colorAttachments()->object(c);
            color->setPixelFormat(pixel_format(pass->attachments[pass->color[c].attachment].format));

            if (info.pColorBlendState && c < info.pColorBlendState->attachmentCount) {
                const VkPipelineColorBlendAttachmentState &blend = info.pColorBlendState->pAttachments[c];
                color->setBlendingEnabled(blend.blendEnable);
                color->setSourceRGBBlendFactor(blend_factor(blend.srcColorBlendFactor));
                color->setDestinationRGBBlendFactor(blend_factor(blend.dstColorBlendFactor));
                color->setSourceAlphaBlendFactor(blend_factor(blend.srcAlphaBlendFactor));
                color->setDestinationAlphaBlendFactor(blend_factor(blend.dstAlphaBlendFactor));
                MTL::ColorWriteMask mask = MTL::ColorWriteMaskNone;
                if (blend.colorWriteMask & VK_COLOR_COMPONENT_R_BIT) mask |= MTL::ColorWriteMaskRed;
                if (blend.colorWriteMask & VK_COLOR_COMPONENT_G_BIT) mask |= MTL::ColorWriteMaskGreen;
                if (blend.colorWriteMask & VK_COLOR_COMPONENT_B_BIT) mask |= MTL::ColorWriteMaskBlue;
                if (blend.colorWriteMask & VK_COLOR_COMPONENT_A_BIT) mask |= MTL::ColorWriteMaskAlpha;
                color->setWriteMask(mask);
            }
        }
        if (pass->has_depth)
            desc->setDepthAttachmentPixelFormat(pixel_format(pass->attachments[pass->depth.attachment].format));

        VkPrimitiveTopology topology = info.pInputAssemblyState ? info.pInputAssemblyState->topology
                                                                : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        switch (topology) {
        case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
            pipeline->primitive = MTL::PrimitiveTypeLine;
            desc->setInputPrimitiveTopology(MTL::PrimitiveTopologyClassLine);
            break;
        case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
            pipeline->primitive = MTL::PrimitiveTypeTriangleStrip;
            desc->setInputPrimitiveTopology(MTL::PrimitiveTopologyClassTriangle);
            break;
        default:
            pipeline->primitive = MTL::PrimitiveTypeTriangle;
            desc->setInputPrimitiveTopology(MTL::PrimitiveTopologyClassTriangle);
            break;
        }

        pipeline->cull = MTL::CullModeNone;
        pipeline->winding = MTL::WindingClockwise;
        pipeline->fill = MTL::TriangleFillModeFill;
        if (const VkPipelineRasterizationStateCreateInfo *rs = info.pRasterizationState) {
            if (rs->cullMode == VK_CULL_MODE_BACK_BIT)
                pipeline->cull = MTL::CullModeBack;
            else if (rs->cullMode == VK_CULL_MODE_FRONT_BIT)
                pipeline->cull = MTL::CullModeFront;
            pipeline->winding = rs->frontFace == VK_FRONT_FACE_CLOCKWISE ? MTL::WindingClockwise
                                                                         : MTL::WindingCounterClockwise;
            if (rs->polygonMode == VK_POLYGON_MODE_LINE)
                pipeline->fill = MTL::TriangleFillModeLines;
        }

        bool dynamic_viewport = false, dynamic_scissor = false;
        if (info.pDynamicState) {
            for (uint32_t d = 0; d < info.pDynamicState->dynamicStateCount; d++) {
                if (info.pDynamicState->pDynamicStates[d] == VK_DYNAMIC_STATE_VIEWPORT)
                    dynamic_viewport = true;
                if (info.pDynamicState->pDynamicStates[d] == VK_DYNAMIC_STATE_SCISSOR)
                    dynamic_scissor = true;
            }
        }
        if (info.pViewportState) {
            if (!dynamic_viewport && info.pViewportState->pViewports) {
                pipeline->has_viewport = true;
                pipeline->viewport = info.pViewportState->pViewports[0];
            }
            if (!dynamic_scissor && info.pViewportState->pScissors) {
                pipeline->has_scissor = true;
                pipeline->scissor = info.pViewportState->pScissors[0];
            }
        }

        if (pass->has_depth) {
            MTL::DepthStencilDescriptor *ds = MTL::DepthStencilDescriptor::alloc()->init();
            const VkPipelineDepthStencilStateCreateInfo *state = info.pDepthStencilState;
            if (state && state->depthTestEnable) {
                ds->setDepthCompareFunction(compare_function(state->depthCompareOp));
                ds->setDepthWriteEnabled(state->depthWriteEnable);
            }
            pipeline->depth_stencil = g.device->newDepthStencilState(ds);
            ds->release();
        }

        if (ok) {
            NS::Error *error = nullptr;
            MTL::RenderPipelineReflection *reflection = nullptr;
            pipeline->render = g.device->newRenderPipelineState(
                desc, MTL::PipelineOptionBindingInfo | MTL::PipelineOptionBufferTypeInfo, &reflection, &error);
            if (pipeline->render) {
                if (pipeline->stages[0].module)
                    pipeline->stages[0].reflect(reflection->vertexBindings());
                if (pipeline->stages[1].module)
                    pipeline->stages[1].reflect(reflection->fragmentBindings());
            } else {
                vkmtl_error("creating a render pipeline failed: %s\n", error_text(error));
                ok = false;
            }
        }
        desc->release();

        if (!ok) {
            delete pipeline;
            result = VK_ERROR_INITIALIZATION_FAILED;
            continue;
        }
        out[i] = reinterpret_cast<VkPipeline>(pipeline);
    }
    return result;
}

void vkDestroyPipeline(VkDevice, VkPipeline handle, const VkAllocationCallbacks *)
{
    Pipeline *pipeline = VKMTL_HANDLE(Pipeline, handle);
    if (!pipeline)
        return;
    if (pipeline->compute)
        pipeline->compute->release();
    if (pipeline->render)
        pipeline->render->release();
    if (pipeline->depth_stencil)
        pipeline->depth_stencil->release();
    delete pipeline;
}

// ------------------------------------------------------------------------------ render passes

VkResult vkCreateRenderPass(VkDevice, const VkRenderPassCreateInfo *info, const VkAllocationCallbacks *, VkRenderPass *out)
{
    RenderPass *pass = new RenderPass();
    pass->attachments.assign(info->pAttachments, info->pAttachments + info->attachmentCount);

    const VkSubpassDescription &subpass = info->pSubpasses[0];
    pass->color.assign(subpass.pColorAttachments, subpass.pColorAttachments + subpass.colorAttachmentCount);
    if (subpass.pDepthStencilAttachment && subpass.pDepthStencilAttachment->attachment != VK_ATTACHMENT_UNUSED) {
        pass->has_depth = true;
        pass->depth = *subpass.pDepthStencilAttachment;
    }
    *out = reinterpret_cast<VkRenderPass>(pass);
    return VK_SUCCESS;
}

void vkDestroyRenderPass(VkDevice, VkRenderPass handle, const VkAllocationCallbacks *)
{
    delete VKMTL_HANDLE(RenderPass, handle);
}

VkResult vkCreateFramebuffer(VkDevice, const VkFramebufferCreateInfo *info, const VkAllocationCallbacks *, VkFramebuffer *out)
{
    Framebuffer *fb = new Framebuffer();
    for (uint32_t i = 0; i < info->attachmentCount; i++)
        fb->views.push_back(VKMTL_HANDLE(ImageView, info->pAttachments[i]));
    fb->width = info->width;
    fb->height = info->height;
    *out = reinterpret_cast<VkFramebuffer>(fb);
    return VK_SUCCESS;
}

void vkDestroyFramebuffer(VkDevice, VkFramebuffer handle, const VkAllocationCallbacks *)
{
    delete VKMTL_HANDLE(Framebuffer, handle);
}

}  // extern "C"
