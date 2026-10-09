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

// vk_metal_commands.cpp -- command buffers and acceleration structures.
//
// Commands are encoded into a Metal command buffer as vkpt records them. Metal needs
// an encoder per kind of work, so the command buffer keeps one open and switches when
// the kind changes. Pipeline barriers and image layouts have no counterpart: a command
// buffer's passes run in order, and the queue runs command buffers in commit order.

#include "vk_metal_internal.hpp"

#include <stdlib.h>
#include <string.h>

namespace vkmtl {

void CommandBuffer::end_encoder()
{
    if (compute) {
        compute->endEncoding();
        compute->release();
        compute = nullptr;
    }
    if (blit) {
        blit->endEncoding();
        blit->release();
        blit = nullptr;
    }
    if (render) {
        render->endEncoding();
        render->release();
        render = nullptr;
    }
}

MTL::ComputeCommandEncoder *CommandBuffer::compute_encoder()
{
    if (!compute) {
        end_encoder();
        compute = cmd->computeCommandEncoder()->retain();
    }
    return compute;
}

MTL::BlitCommandEncoder *CommandBuffer::blit_encoder()
{
    if (!blit) {
        end_encoder();
        blit = cmd->blitCommandEncoder()->retain();
    }
    return blit;
}

// ------------------------------------------------------------------------------ scaled blit

static const char *s_blit_source = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct BlitOut {
    float4 position [[position]];
    float2 uv;
};

struct BlitRect {
    float2 uv_min;
    float2 uv_max;
};

vertex BlitOut blit_vs(uint vid [[vertex_id]], constant BlitRect &rect [[buffer(0)]])
{
    float2 corner = float2(float((vid << 1) & 2), float(vid & 2));
    BlitOut out;
    out.position = float4(corner * 2.0 - 1.0, 0.0, 1.0);
    out.uv = mix(rect.uv_min, rect.uv_max, float2(corner.x, 1.0 - corner.y));
    return out;
}

fragment float4 blit_fs(BlitOut in [[stage_in]], texture2d<float> tex [[texture(0)]], sampler smp [[sampler(0)]])
{
    return tex.sample(smp, in.uv);
}
)MSL";

static MTL::RenderPipelineState *blit_pipeline(MTL::PixelFormat format)
{
    static MTL::Library *library;
    static std::map<MTL::PixelFormat, MTL::RenderPipelineState *> pipelines;

    auto it = pipelines.find(format);
    if (it != pipelines.end())
        return it->second;

    NS::Error *error = nullptr;
    if (!library) {
        library = g.device->newLibrary(nsstr(s_blit_source), nullptr, &error);
        if (!library) {
            vkmtl_error("compiling the blit shader failed\n");
            return nullptr;
        }
    }

    MTL::Function *vs = library->newFunction(nsstr("blit_vs"));
    MTL::Function *fs = library->newFunction(nsstr("blit_fs"));
    MTL::RenderPipelineDescriptor *desc = MTL::RenderPipelineDescriptor::alloc()->init();
    desc->setVertexFunction(vs);
    desc->setFragmentFunction(fs);
    desc->colorAttachments()->object(0)->setPixelFormat(format);
    MTL::RenderPipelineState *state = g.device->newRenderPipelineState(desc, &error);
    desc->release();
    vs->release();
    fs->release();
    if (!state)
        vkmtl_error("creating the blit pipeline failed\n");
    pipelines[format] = state;
    return state;
}

void blit_scaled(CommandBuffer *cb, MTL::Texture *src, uint32_t src_level, uint32_t src_slice,
                 const VkOffset3D src_offsets[2], MTL::Texture *dst, uint32_t dst_level,
                 uint32_t dst_slice, const VkOffset3D dst_offsets[2], bool linear)
{
    static MTL::SamplerState *samplers[2];
    if (!samplers[0]) {
        for (int i = 0; i < 2; i++) {
            MTL::SamplerDescriptor *sd = MTL::SamplerDescriptor::alloc()->init();
            sd->setMinFilter(i ? MTL::SamplerMinMagFilterLinear : MTL::SamplerMinMagFilterNearest);
            sd->setMagFilter(i ? MTL::SamplerMinMagFilterLinear : MTL::SamplerMinMagFilterNearest);
            samplers[i] = g.device->newSamplerState(sd);
            sd->release();
        }
    }

    MTL::RenderPipelineState *pipeline = blit_pipeline(dst->pixelFormat());
    if (!pipeline)
        return;

    cb->end_encoder();

    // A view of just the source level, so reading one level of a texture while
    // writing another (mipmap generation) is well defined.
    MTL::Texture *view = src->newTextureView(src->pixelFormat(), MTL::TextureType2D,
                                             NS::Range::Make(src_level, 1), NS::Range::Make(src_slice, 1));

    MTL::RenderPassDescriptor *pass = MTL::RenderPassDescriptor::renderPassDescriptor();
    MTL::RenderPassColorAttachmentDescriptor *color = pass->colorAttachments()->object(0);
    color->setTexture(dst);
    color->setLevel(dst_level);
    color->setSlice(dst_slice);
    color->setLoadAction(MTL::LoadActionLoad);
    color->setStoreAction(MTL::StoreActionStore);

    float src_width = (float)view->width(), src_height = (float)view->height();
    float rect[4] = {
        src_offsets[0].x / src_width, src_offsets[0].y / src_height,
        src_offsets[1].x / src_width, src_offsets[1].y / src_height,
    };

    MTL::Viewport viewport = {
        (double)dst_offsets[0].x, (double)dst_offsets[0].y,
        (double)(dst_offsets[1].x - dst_offsets[0].x), (double)(dst_offsets[1].y - dst_offsets[0].y), 0.0, 1.0
    };

    MTL::RenderCommandEncoder *enc = cb->cmd->renderCommandEncoder(pass);
    enc->setRenderPipelineState(pipeline);
    enc->setViewport(viewport);
    enc->setVertexBytes(rect, sizeof(rect), 0);
    enc->setFragmentTexture(view, 0);
    enc->setFragmentSamplerState(samplers[linear ? 1 : 0], 0);
    enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    enc->endEncoding();
    view->release();
}

}  // namespace vkmtl

using namespace vkmtl;

namespace {

int bind_index(VkPipelineBindPoint bind_point)
{
    return bind_point == VK_PIPELINE_BIND_POINT_COMPUTE ? 1 : 0;
}

template <typename F>
void bind_stage_sets(CommandBuffer *cb, int bp, const Stage &stage, F set_buffer)
{
    for (uint32_t index = 0; index < MAX_SETS; index++) {
        if (!stage.sets[index])
            continue;
        uint32_t source = index >= SAMPLER_SET_OFFSET ? index - SAMPLER_SET_OFFSET : index;
        DescSet *set = cb->sets[bp][source];
        if (!set) {
            vkmtl_error("descriptor set %u is not bound\n", source);
            continue;
        }
        set_buffer(set->encode(stage.sets[index]), index);
    }
}

// Sets pipeline state, descriptor sets and push constants on the render encoder.
bool prepare_draw(CommandBuffer *cb)
{
    Pipeline *pipeline = cb->pipelines[0];
    MTL::RenderCommandEncoder *enc = cb->render;
    if (!pipeline || !pipeline->render || !enc)
        return false;

    enc->setRenderPipelineState(pipeline->render);
    if (pipeline->depth_stencil)
        enc->setDepthStencilState(pipeline->depth_stencil);
    enc->setCullMode(pipeline->cull);
    enc->setFrontFacingWinding(pipeline->winding);
    enc->setTriangleFillMode(pipeline->fill);

    VkViewport vp = { 0, 0, (float)cb->target_width, (float)cb->target_height, 0, 1 };
    if (pipeline->has_viewport)
        vp = pipeline->viewport;
    else if (cb->has_viewport)
        vp = cb->viewport;
    enc->setViewport(MTL::Viewport{ vp.x, vp.y, vp.width, vp.height, vp.minDepth, vp.maxDepth });

    VkRect2D sc = { { 0, 0 }, { cb->target_width, cb->target_height } };
    if (pipeline->has_scissor)
        sc = pipeline->scissor;
    else if (cb->has_scissor)
        sc = cb->scissor;
    int32_t x0 = sc.offset.x < 0 ? 0 : sc.offset.x;
    int32_t y0 = sc.offset.y < 0 ? 0 : sc.offset.y;
    int64_t x1 = (int64_t)sc.offset.x + sc.extent.width, y1 = (int64_t)sc.offset.y + sc.extent.height;
    if (x1 > cb->target_width) x1 = cb->target_width;
    if (y1 > cb->target_height) y1 = cb->target_height;
    if (x1 <= x0 || y1 <= y0)
        return false;
    enc->setScissorRect(MTL::ScissorRect{ (NS::UInteger)x0, (NS::UInteger)y0, (NS::UInteger)(x1 - x0), (NS::UInteger)(y1 - y0) });

    bind_stage_sets(cb, 0, pipeline->stages[0], [enc](MTL::Buffer *b, uint32_t i) { enc->setVertexBuffer(b, 0, i); });
    bind_stage_sets(cb, 0, pipeline->stages[1], [enc](MTL::Buffer *b, uint32_t i) { enc->setFragmentBuffer(b, 0, i); });
    if (pipeline->stages[0].has_push)
        enc->setVertexBytes(cb->push, sizeof(cb->push), PUSH_CONSTANT_BUFFER_INDEX);
    if (pipeline->stages[1].has_push)
        enc->setFragmentBytes(cb->push, sizeof(cb->push), PUSH_CONSTANT_BUFFER_INDEX);

    for (int i = 0; i < MAX_VERTEX_BINDINGS; i++) {
        if (cb->vertex_buffers[i])
            enc->setVertexBuffer(cb->vertex_buffers[i]->mtl(), cb->vertex_buffers[i]->offset + cb->vertex_offsets[i],
                                 VERTEX_BUFFER_BASE_INDEX + i);
    }
    return true;
}

MTL::AccelerationStructureDescriptor *accel_descriptor(const VkAccelerationStructureBuildGeometryInfoKHR *info,
                                                       const uint32_t *counts,
                                                       const VkAccelerationStructureBuildRangeInfoKHR *ranges,
                                                       bool for_build)
{
    MTL::AccelerationStructureUsage usage = MTL::AccelerationStructureUsageNone;
    if (info->flags & VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR)
        usage |= MTL::AccelerationStructureUsagePreferFastBuild;

    if (info->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
        const VkAccelerationStructureGeometryKHR &geom = info->pGeometries ? info->pGeometries[0] : *info->ppGeometries[0];
        uint32_t count = ranges ? ranges[0].primitiveCount : counts[0];

        MTL::InstanceAccelerationStructureDescriptor *desc = MTL::InstanceAccelerationStructureDescriptor::descriptor();
        desc->setUsage(usage);
        desc->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeIndirect);
        desc->setInstanceCount(count);

        if (for_build && count) {
            MTL::Buffer *src_buffer;
            size_t src_offset;
            if (!resolve_address(geom.geometry.instances.data.deviceAddress, &src_buffer, &src_offset)) {
                vkmtl_error("instance data address is not in a buffer\n");
                return nullptr;
            }
            const VkAccelerationStructureInstanceKHR *src =
                (const VkAccelerationStructureInstanceKHR *)((const char *)src_buffer->contents() + src_offset);

            MTL::Buffer *buffer = g.device->newBuffer(count * sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor),
                                                      MTL::ResourceStorageModeShared);
            auto *dst = static_cast<MTL::IndirectAccelerationStructureInstanceDescriptor *>(buffer->contents());

            for (uint32_t i = 0; i < count; i++) {
                const VkAccelerationStructureInstanceKHR &in = src[i];
                MTL::IndirectAccelerationStructureInstanceDescriptor &out = dst[i];
                memset(&out, 0, sizeof(out));

                // row-major 3x4 to four columns
                for (int c = 0; c < 4; c++)
                    out.transformationMatrix.columns[c] = MTL::PackedFloat3(
                        in.transform.matrix[0][c], in.transform.matrix[1][c], in.transform.matrix[2][c]);

                MTL::AccelerationStructureInstanceOptions options = MTL::AccelerationStructureInstanceOptionNone;
                if (in.flags & VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR)
                    options |= MTL::AccelerationStructureInstanceOptionDisableTriangleCulling;
                if (in.flags & VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR)
                    options |= MTL::AccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise;
                if (in.flags & VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR)
                    options |= MTL::AccelerationStructureInstanceOptionOpaque;
                if (in.flags & VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR)
                    options |= MTL::AccelerationStructureInstanceOptionNonOpaque;
                out.options = options;
                out.mask = in.mask;

                // There is no shader binding table: the shaders read the hit group
                // offset from the top bits of the user ID (see path_tracer_rgen.h).
                out.userID = in.instanceCustomIndex | (in.instanceShaderBindingTableRecordOffset << 28);

                Accel *blas = reinterpret_cast<Accel *>(in.accelerationStructureReference);
                if (blas)
                    out.accelerationStructureID = blas->as->gpuResourceID();
                else
                    out.mask = 0;
            }
            desc->setInstanceDescriptorBuffer(buffer);
            buffer->release();  // the descriptor and then the command buffer keep it
        }
        return desc;
    }

    std::vector<NS::Object *> geometries;
    for (uint32_t i = 0; i < info->geometryCount; i++) {
        const VkAccelerationStructureGeometryKHR &geom = info->pGeometries ? info->pGeometries[i] : *info->ppGeometries[i];
        uint32_t count = ranges ? ranges[i].primitiveCount : counts[i];
        size_t data_offset = ranges ? ranges[i].primitiveOffset : 0;
        bool opaque = (geom.flags & VK_GEOMETRY_OPAQUE_BIT_KHR) != 0;

        if (geom.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
            const VkAccelerationStructureGeometryTrianglesDataKHR &tri = geom.geometry.triangles;
            MTL::AccelerationStructureTriangleGeometryDescriptor *gd =
                MTL::AccelerationStructureTriangleGeometryDescriptor::descriptor();
            gd->setTriangleCount(count);
            gd->setVertexStride(tri.vertexStride);
            gd->setVertexFormat(MTL::AttributeFormatFloat3);
            gd->setOpaque(opaque);
            bool indexed = tri.indexType != VK_INDEX_TYPE_NONE_KHR;
            if (indexed)
                gd->setIndexType(tri.indexType == VK_INDEX_TYPE_UINT16 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32);

            if (for_build) {
                MTL::Buffer *buffer;
                size_t offset;
                if (!resolve_address(tri.vertexData.deviceAddress, &buffer, &offset)) {
                    vkmtl_error("vertex data address is not in a buffer\n");
                    return nullptr;
                }
                gd->setVertexBuffer(buffer);
                gd->setVertexBufferOffset(offset + (indexed ? 0 : data_offset));
                if (indexed) {
                    if (!resolve_address(tri.indexData.deviceAddress, &buffer, &offset)) {
                        vkmtl_error("index data address is not in a buffer\n");
                        return nullptr;
                    }
                    gd->setIndexBuffer(buffer);
                    gd->setIndexBufferOffset(offset + data_offset);
                }
            }
            geometries.push_back(gd);
        } else if (geom.geometryType == VK_GEOMETRY_TYPE_AABBS_KHR) {
            const VkAccelerationStructureGeometryAabbsDataKHR &aabbs = geom.geometry.aabbs;
            MTL::AccelerationStructureBoundingBoxGeometryDescriptor *gd =
                MTL::AccelerationStructureBoundingBoxGeometryDescriptor::descriptor();
            gd->setBoundingBoxCount(count);
            gd->setBoundingBoxStride(aabbs.stride);
            gd->setOpaque(opaque);
            if (for_build) {
                MTL::Buffer *buffer;
                size_t offset;
                if (!resolve_address(aabbs.data.deviceAddress, &buffer, &offset)) {
                    vkmtl_error("bounding box data address is not in a buffer\n");
                    return nullptr;
                }
                gd->setBoundingBoxBuffer(buffer);
                gd->setBoundingBoxBufferOffset(offset + data_offset);
            }
            geometries.push_back(gd);
        }
    }

    MTL::PrimitiveAccelerationStructureDescriptor *desc = MTL::PrimitiveAccelerationStructureDescriptor::descriptor();
    desc->setUsage(usage);
    desc->setGeometryDescriptors(NS::Array::array(geometries.data(), geometries.size()));
    return desc;
}

}  // namespace

extern "C" {

// ------------------------------------------------------------------------------ command buffers

VkResult vkCreateCommandPool(VkDevice, const VkCommandPoolCreateInfo *, const VkAllocationCallbacks *, VkCommandPool *out)
{
    static int dummy;
    *out = reinterpret_cast<VkCommandPool>(&dummy);
    return VK_SUCCESS;
}

void vkDestroyCommandPool(VkDevice, VkCommandPool, const VkAllocationCallbacks *)
{
}

VkResult vkAllocateCommandBuffers(VkDevice, const VkCommandBufferAllocateInfo *info, VkCommandBuffer *out)
{
    for (uint32_t i = 0; i < info->commandBufferCount; i++)
        out[i] = reinterpret_cast<VkCommandBuffer>(new CommandBuffer());
    return VK_SUCCESS;
}

static void discard(CommandBuffer *cb)
{
    cb->end_encoder();
    if (cb->cmd) {
        cb->cmd->release();
        cb->cmd = nullptr;
    }
}

void vkFreeCommandBuffers(VkDevice, VkCommandPool, uint32_t count, const VkCommandBuffer *buffers)
{
    for (uint32_t i = 0; i < count; i++) {
        CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, buffers[i]);
        if (!cb)
            continue;
        discard(cb);
        delete cb;
    }
}

VkResult vkResetCommandBuffer(VkCommandBuffer handle, VkCommandBufferResetFlags)
{
    discard(VKMTL_HANDLE(CommandBuffer, handle));
    return VK_SUCCESS;
}

VkResult vkBeginCommandBuffer(VkCommandBuffer handle, const VkCommandBufferBeginInfo *)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    discard(cb);
    *cb = CommandBuffer();
    cb->cmd = g.queue->commandBuffer()->retain();
    return VK_SUCCESS;
}

VkResult vkEndCommandBuffer(VkCommandBuffer handle)
{
    VKMTL_HANDLE(CommandBuffer, handle)->end_encoder();
    return VK_SUCCESS;
}

VkResult vkQueueSubmit(VkQueue, uint32_t count, const VkSubmitInfo *submits, VkFence fence_handle)
{
    Fence *fence = VKMTL_HANDLE(Fence, fence_handle);
    MTL::CommandBuffer *last = nullptr;

    residency_commit();

    for (uint32_t s = 0; s < count; s++) {
        for (uint32_t i = 0; i < submits[s].commandBufferCount; i++) {
            CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, submits[s].pCommandBuffers[i]);
            if (!cb->cmd)
                continue;
            cb->end_encoder();
            cb->cmd->addCompletedHandler([](MTL::CommandBuffer *done) {
                if (done->status() == MTL::CommandBufferStatusError) {
                    NS::Error *error = done->error();
                    vkmtl_error("command buffer failed: %s\n",
                                error && error->localizedDescription() ? error->localizedDescription()->utf8String() : "unknown error");
                }
            });
            cb->cmd->commit();
            submitted(cb->cmd);
            last = cb->cmd;
            cb->cmd->release();
            cb->cmd = nullptr;
        }
    }

    if (fence) {
        if (fence->pending)
            fence->pending->release();
        fence->pending = last ? last->retain() : nullptr;
        fence->signaled = !last;
    }
    return VK_SUCCESS;
}

// ------------------------------------------------------------------------------ state

void vkCmdBindPipeline(VkCommandBuffer handle, VkPipelineBindPoint bind_point, VkPipeline pipeline)
{
    VKMTL_HANDLE(CommandBuffer, handle)->pipelines[bind_index(bind_point)] = VKMTL_HANDLE(Pipeline, pipeline);
}

void vkCmdBindDescriptorSets(VkCommandBuffer handle, VkPipelineBindPoint bind_point, VkPipelineLayout, uint32_t first,
                             uint32_t count, const VkDescriptorSet *sets, uint32_t, const uint32_t *)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    for (uint32_t i = 0; i < count && first + i < MAX_SETS; i++)
        cb->sets[bind_index(bind_point)][first + i] = VKMTL_HANDLE(DescSet, sets[i]);
}

void vkCmdPushConstants(VkCommandBuffer handle, VkPipelineLayout, VkShaderStageFlags, uint32_t offset, uint32_t size,
                        const void *values)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    if (offset + size <= sizeof(cb->push))
        memcpy(cb->push + offset, values, size);
}

void vkCmdSetViewport(VkCommandBuffer handle, uint32_t, uint32_t, const VkViewport *viewports)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    cb->viewport = viewports[0];
    cb->has_viewport = true;
}

void vkCmdSetScissor(VkCommandBuffer handle, uint32_t, uint32_t, const VkRect2D *scissors)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    cb->scissor = scissors[0];
    cb->has_scissor = true;
}

void vkCmdSetLineWidth(VkCommandBuffer, float)
{
}

void vkCmdSetDeviceMask(VkCommandBuffer, uint32_t)
{
}

void vkCmdBindVertexBuffers(VkCommandBuffer handle, uint32_t first, uint32_t count, const VkBuffer *buffers,
                            const VkDeviceSize *offsets)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    for (uint32_t i = 0; i < count && first + i < MAX_VERTEX_BINDINGS; i++) {
        cb->vertex_buffers[first + i] = VKMTL_HANDLE(Buffer, buffers[i]);
        cb->vertex_offsets[first + i] = offsets[i];
    }
}

void vkCmdBindIndexBuffer(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset, VkIndexType type)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    cb->index_buffer = VKMTL_HANDLE(Buffer, buffer);
    cb->index_offset = offset;
    cb->index_type = type;
}

void vkCmdPipelineBarrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, uint32_t,
                          const VkMemoryBarrier *, uint32_t, const VkBufferMemoryBarrier *, uint32_t,
                          const VkImageMemoryBarrier *)
{
}

void vkCmdWriteTimestamp(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, uint32_t)
{
}

void vkCmdResetQueryPool(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t)
{
}

// ------------------------------------------------------------------------------ compute

void vkCmdDispatch(VkCommandBuffer handle, uint32_t x, uint32_t y, uint32_t z)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    Pipeline *pipeline = cb->pipelines[1];
    if (!pipeline || !pipeline->compute || !x || !y || !z)
        return;

    MTL::ComputeCommandEncoder *enc = cb->compute_encoder();
    enc->setComputePipelineState(pipeline->compute);
    bind_stage_sets(cb, 1, pipeline->stages[0], [enc](MTL::Buffer *b, uint32_t i) { enc->setBuffer(b, 0, i); });
    if (pipeline->stages[0].has_push)
        enc->setBytes(cb->push, sizeof(cb->push), PUSH_CONSTANT_BUFFER_INDEX);
    enc->dispatchThreadgroups(MTL::Size::Make(x, y, z), pipeline->threads_per_group);
}

// ------------------------------------------------------------------------------ rendering

void vkCmdBeginRenderPass(VkCommandBuffer handle, const VkRenderPassBeginInfo *info, VkSubpassContents)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    RenderPass *rp = VKMTL_HANDLE(RenderPass, info->renderPass);
    Framebuffer *fb = VKMTL_HANDLE(Framebuffer, info->framebuffer);

    cb->end_encoder();

    auto load_action = [](VkAttachmentLoadOp op) {
        return op == VK_ATTACHMENT_LOAD_OP_CLEAR ? MTL::LoadActionClear
             : op == VK_ATTACHMENT_LOAD_OP_LOAD ? MTL::LoadActionLoad : MTL::LoadActionDontCare;
    };

    MTL::RenderPassDescriptor *pass = MTL::RenderPassDescriptor::renderPassDescriptor();
    for (size_t c = 0; c < rp->color.size(); c++) {
        uint32_t index = rp->color[c].attachment;
        MTL::RenderPassColorAttachmentDescriptor *color = pass->colorAttachments()->object(c);
        color->setTexture(fb->views[index]->texture);
        color->setLoadAction(load_action(rp->attachments[index].loadOp));
        color->setStoreAction(MTL::StoreActionStore);
        if (index < info->clearValueCount) {
            const float *v = info->pClearValues[index].color.float32;
            color->setClearColor(MTL::ClearColor::Make(v[0], v[1], v[2], v[3]));
        }
    }
    if (rp->has_depth) {
        uint32_t index = rp->depth.attachment;
        MTL::RenderPassDepthAttachmentDescriptor *depth = pass->depthAttachment();
        depth->setTexture(fb->views[index]->texture);
        depth->setLoadAction(load_action(rp->attachments[index].loadOp));
        depth->setStoreAction(MTL::StoreActionStore);
        if (index < info->clearValueCount)
            depth->setClearDepth(info->pClearValues[index].depthStencil.depth);
    }

    cb->render = cb->cmd->renderCommandEncoder(pass)->retain();
    cb->target_width = fb->width;
    cb->target_height = fb->height;
}

void vkCmdEndRenderPass(VkCommandBuffer handle)
{
    VKMTL_HANDLE(CommandBuffer, handle)->end_encoder();
}

void vkCmdDraw(VkCommandBuffer handle, uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
               uint32_t first_instance)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    if (!vertex_count || !instance_count || !prepare_draw(cb))
        return;
    cb->render->drawPrimitives(cb->pipelines[0]->primitive, (NS::UInteger)first_vertex, (NS::UInteger)vertex_count,
                               (NS::UInteger)instance_count, (NS::UInteger)first_instance);
}

void vkCmdDrawIndexed(VkCommandBuffer handle, uint32_t index_count, uint32_t instance_count, uint32_t first_index,
                      int32_t vertex_offset, uint32_t first_instance)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    if (!index_count || !instance_count || !cb->index_buffer || !prepare_draw(cb))
        return;
    bool wide = cb->index_type == VK_INDEX_TYPE_UINT32;
    cb->render->drawIndexedPrimitives(cb->pipelines[0]->primitive, index_count,
                                      wide ? MTL::IndexTypeUInt32 : MTL::IndexTypeUInt16, cb->index_buffer->mtl(),
                                      cb->index_buffer->offset + cb->index_offset + (size_t)first_index * (wide ? 4 : 2),
                                      instance_count, vertex_offset, first_instance);
}

// ------------------------------------------------------------------------------ copies

void vkCmdCopyBuffer(VkCommandBuffer handle, VkBuffer src_handle, VkBuffer dst_handle, uint32_t count,
                     const VkBufferCopy *regions)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    Buffer *src = VKMTL_HANDLE(Buffer, src_handle), *dst = VKMTL_HANDLE(Buffer, dst_handle);
    MTL::BlitCommandEncoder *blit = cb->blit_encoder();
    for (uint32_t i = 0; i < count; i++) {
        if (regions[i].size)
            blit->copyFromBuffer(src->mtl(), src->offset + regions[i].srcOffset, dst->mtl(),
                                 dst->offset + regions[i].dstOffset, regions[i].size);
    }
}

void vkCmdFillBuffer(VkCommandBuffer handle, VkBuffer buffer_handle, VkDeviceSize offset, VkDeviceSize size, uint32_t data)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    Buffer *buffer = VKMTL_HANDLE(Buffer, buffer_handle);
    if (size == VK_WHOLE_SIZE)
        size = buffer->size - offset;

    uint8_t byte = data & 0xff;
    if (data != byte * 0x01010101u) {
        vkmtl_unsupported("filling a buffer with a non-repeating value");
        return;
    }
    cb->blit_encoder()->fillBuffer(buffer->mtl(), NS::Range::Make(buffer->offset + offset, size), byte);
}

void vkCmdCopyBufferToImage(VkCommandBuffer handle, VkBuffer src_handle, VkImage dst_handle, VkImageLayout,
                            uint32_t count, const VkBufferImageCopy *regions)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    Buffer *src = VKMTL_HANDLE(Buffer, src_handle);
    Image *dst = VKMTL_HANDLE(Image, dst_handle);
    uint32_t bpp = format_bytes_per_pixel(dst->info.format);
    MTL::BlitCommandEncoder *blit = cb->blit_encoder();

    for (uint32_t i = 0; i < count; i++) {
        const VkBufferImageCopy &r = regions[i];
        size_t row = (size_t)(r.bufferRowLength ? r.bufferRowLength : r.imageExtent.width) * bpp;
        size_t image = row * (r.bufferImageHeight ? r.bufferImageHeight : r.imageExtent.height);
        for (uint32_t layer = 0; layer < r.imageSubresource.layerCount; layer++) {
            blit->copyFromBuffer(src->mtl(), src->offset + r.bufferOffset + layer * image * r.imageExtent.depth, row, image,
                                 MTL::Size::Make(r.imageExtent.width, r.imageExtent.height, r.imageExtent.depth),
                                 dst->texture, r.imageSubresource.baseArrayLayer + layer, r.imageSubresource.mipLevel,
                                 MTL::Origin::Make(r.imageOffset.x, r.imageOffset.y, r.imageOffset.z));
        }
    }
}

void vkCmdCopyImage(VkCommandBuffer handle, VkImage src_handle, VkImageLayout, VkImage dst_handle, VkImageLayout,
                    uint32_t count, const VkImageCopy *regions)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    Image *src = VKMTL_HANDLE(Image, src_handle), *dst = VKMTL_HANDLE(Image, dst_handle);
    MTL::BlitCommandEncoder *blit = cb->blit_encoder();

    for (uint32_t i = 0; i < count; i++) {
        const VkImageCopy &r = regions[i];
        for (uint32_t layer = 0; layer < r.srcSubresource.layerCount; layer++) {
            blit->copyFromTexture(src->texture, r.srcSubresource.baseArrayLayer + layer, r.srcSubresource.mipLevel,
                                  MTL::Origin::Make(r.srcOffset.x, r.srcOffset.y, r.srcOffset.z),
                                  MTL::Size::Make(r.extent.width, r.extent.height, r.extent.depth),
                                  dst->texture, r.dstSubresource.baseArrayLayer + layer, r.dstSubresource.mipLevel,
                                  MTL::Origin::Make(r.dstOffset.x, r.dstOffset.y, r.dstOffset.z));
        }
    }
}

void vkCmdBlitImage(VkCommandBuffer handle, VkImage src_handle, VkImageLayout, VkImage dst_handle, VkImageLayout,
                    uint32_t count, const VkImageBlit *regions, VkFilter filter)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    Image *src = VKMTL_HANDLE(Image, src_handle), *dst = VKMTL_HANDLE(Image, dst_handle);

    for (uint32_t i = 0; i < count; i++) {
        const VkImageBlit &r = regions[i];
        for (uint32_t layer = 0; layer < r.srcSubresource.layerCount; layer++) {
            blit_scaled(cb, src->texture, r.srcSubresource.mipLevel, r.srcSubresource.baseArrayLayer + layer,
                        r.srcOffsets, dst->texture, r.dstSubresource.mipLevel,
                        r.dstSubresource.baseArrayLayer + layer, r.dstOffsets, filter == VK_FILTER_LINEAR);
        }
    }
}

void vkCmdClearColorImage(VkCommandBuffer handle, VkImage image_handle, VkImageLayout, const VkClearColorValue *color,
                          uint32_t count, const VkImageSubresourceRange *ranges)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    Image *image = VKMTL_HANDLE(Image, image_handle);
    cb->end_encoder();

    for (uint32_t i = 0; i < count; i++) {
        uint32_t levels = ranges[i].levelCount == VK_REMAINING_MIP_LEVELS
            ? image->info.mipLevels - ranges[i].baseMipLevel : ranges[i].levelCount;
        uint32_t layers = ranges[i].layerCount == VK_REMAINING_ARRAY_LAYERS
            ? image->info.arrayLayers - ranges[i].baseArrayLayer : ranges[i].layerCount;

        for (uint32_t level = 0; level < levels; level++) {
            for (uint32_t layer = 0; layer < layers; layer++) {
                MTL::RenderPassDescriptor *pass = MTL::RenderPassDescriptor::renderPassDescriptor();
                MTL::RenderPassColorAttachmentDescriptor *att = pass->colorAttachments()->object(0);
                att->setTexture(image->texture);
                att->setLevel(ranges[i].baseMipLevel + level);
                att->setSlice(ranges[i].baseArrayLayer + layer);
                att->setLoadAction(MTL::LoadActionClear);
                att->setStoreAction(MTL::StoreActionStore);
                att->setClearColor(MTL::ClearColor::Make(color->float32[0], color->float32[1], color->float32[2],
                                                         color->float32[3]));
                cb->cmd->renderCommandEncoder(pass)->endEncoding();
            }
        }
    }
}

// ------------------------------------------------------------------------------ acceleration structures

void vkGetAccelerationStructureBuildSizesKHR(VkDevice, VkAccelerationStructureBuildTypeKHR,
                                             const VkAccelerationStructureBuildGeometryInfoKHR *info,
                                             const uint32_t *counts, VkAccelerationStructureBuildSizesInfoKHR *sizes)
{
    MTL::AccelerationStructureDescriptor *desc = accel_descriptor(info, counts, nullptr, false);
    MTL::AccelerationStructureSizes s = g.device->accelerationStructureSizes(desc);
    sizes->accelerationStructureSize = s.accelerationStructureSize;
    sizes->buildScratchSize = s.buildScratchBufferSize;
    sizes->updateScratchSize = s.refitScratchBufferSize;
}

VkResult vkCreateAccelerationStructureKHR(VkDevice, const VkAccelerationStructureCreateInfoKHR *info,
                                          const VkAllocationCallbacks *, VkAccelerationStructureKHR *out)
{
    Accel *accel = new Accel();
    accel->type = info->type;
    accel->as = g.device->newAccelerationStructure(info->size);
    if (!accel->as) {
        vkmtl_error("couldn't create a %llu byte acceleration structure\n", (unsigned long long)info->size);
        delete accel;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    residency_add(accel->as);
    *out = reinterpret_cast<VkAccelerationStructureKHR>(accel);
    return VK_SUCCESS;
}

void vkDestroyAccelerationStructureKHR(VkDevice, VkAccelerationStructureKHR handle, const VkAllocationCallbacks *)
{
    Accel *accel = VKMTL_HANDLE(Accel, handle);
    if (!accel)
        return;
    residency_remove(accel->as);
    accel->as->release();
    delete accel;
}

VkDeviceAddress vkGetAccelerationStructureDeviceAddressKHR(VkDevice, const VkAccelerationStructureDeviceAddressInfoKHR *info)
{
    // Only ever stored in instance data, which accel_descriptor reads back.
    return reinterpret_cast<VkDeviceAddress>(info->accelerationStructure);
}

void vkCmdBuildAccelerationStructuresKHR(VkCommandBuffer handle, uint32_t count,
                                         const VkAccelerationStructureBuildGeometryInfoKHR *infos,
                                         const VkAccelerationStructureBuildRangeInfoKHR *const *ranges)
{
    CommandBuffer *cb = VKMTL_HANDLE(CommandBuffer, handle);
    if (!count)
        return;
    cb->end_encoder();

    MTL::AccelerationStructureCommandEncoder *enc = cb->cmd->accelerationStructureCommandEncoder();
    for (uint32_t i = 0; i < count; i++) {
        Accel *dst = VKMTL_HANDLE(Accel, infos[i].dstAccelerationStructure);
        MTL::AccelerationStructureDescriptor *desc = accel_descriptor(&infos[i], nullptr, ranges[i], true);
        MTL::Buffer *scratch;
        size_t scratch_offset;
        if (!desc || !dst)
            continue;
        if (!resolve_address(infos[i].scratchData.deviceAddress, &scratch, &scratch_offset)) {
            vkmtl_error("scratch address is not in a buffer\n");
            continue;
        }
        enc->buildAccelerationStructure(dst->as, desc, scratch, scratch_offset);
    }
    enc->endEncoding();
}

}  // extern "C"
