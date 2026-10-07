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

// mtl_device.cpp -- Metal device, textures and frame submission.
//
// Frame model: everything for a frame is recorded in mtl_end_frame. The 2D quads are
// drawn into an offscreen backbuffer which is then copied into the drawable, so the last
// presented image can always be read back for screenshots. Per-frame CPU-written buffers
// are triple-buffered and gated by a semaphore signalled from the command buffer's
// completion handler, so the CPU never overwrites memory the GPU is still reading.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <dispatch/dispatch.h>
#include <string.h>

#include "mtl_api.h"

extern "C" void Com_LPrintf(int type, const char *fmt, ...);
#define MTL_PRINT_ALL     0
#define MTL_PRINT_WARNING 3
#define MTL_PRINT_ERROR   4

#define MAX_QUADS (1 << 14)  // vkpt MAX_STRETCH_PICS

static const char *s_quad_shader_source = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct Quad {
    float x, y, w, h;
    float s, t, w_s, h_t;
    uint color;
    uint tex;
};

struct QuadOut {
    float4 position [[position]];
    float4 color;
    float2 tex_coord;
};

constant float2 corners[4] = { float2(0, 1), float2(0, 0), float2(1, 1), float2(1, 0) };

// Mirrors vkpt stretch_pic.vert: positions are NDC with y down, colors are sRGB-encoded
// and linearized here because the render target is an sRGB format.
vertex QuadOut quad_vs(uint vid [[vertex_id]],
                       uint iid [[instance_id]],
                       const device Quad *quads [[buffer(0)]])
{
    Quad q = quads[iid];
    float2 corner = corners[vid];
    float2 pos = corner * float2(q.w, q.h) + float2(q.x, q.y);

    QuadOut out;
    out.position = float4(pos.x, -pos.y, 0.0, 1.0);
    out.color = pow(unpack_unorm4x8_to_float(q.color), float4(2.4));
    out.tex_coord = float2(q.s, q.t) + corner * float2(q.w_s, q.h_t);
    return out;
}

fragment float4 quad_fs(QuadOut in [[stage_in]],
                        texture2d<float> tex [[texture(0)]],
                        sampler smp [[sampler(0)]])
{
    return in.color * tex.sample(smp, in.tex_coord, level(0));
}
)MSL";

struct texture_slot_t {
    MTL::Texture *texture;
    mtl_filter_t filter;
};

static struct {
    MTL::Device *device;
    MTL::CommandQueue *queue;
    CA::MetalLayer *layer;

    MTL::RenderPipelineState *quad_pipeline;
    MTL::SamplerState *samplers[2];  // indexed by mtl_filter_t

    MTL::Buffer *quad_buffers[MTL_MAX_FRAMES_IN_FLIGHT];
    unsigned frame_index;
    dispatch_semaphore_t in_flight;

    MTL::Texture *backbuffer;
    bool have_presented_frame;

    // Mipmap generation for uploaded textures is batched into one command buffer that
    // is committed ahead of the next frame.
    MTL::CommandBuffer *upload_cmd;
    MTL::BlitCommandEncoder *upload_blit;

    texture_slot_t textures[MTL_MAX_TEXTURES];
    MTL::Texture *white;

    char device_name[256];
} mtl;

static void report_error(const char *what, NS::Error *error)
{
    const char *msg = error && error->localizedDescription()
        ? error->localizedDescription()->utf8String() : "unknown error";
    Com_LPrintf(MTL_PRINT_ERROR, "Metal: %s: %s\n", what, msg);
}

static bool create_quad_pipeline(void)
{
    NS::Error *error = nullptr;
    NS::String *source = NS::String::string(s_quad_shader_source, NS::UTF8StringEncoding);
    MTL::Library *library = mtl.device->newLibrary(source, nullptr, &error);
    if (!library) {
        report_error("compiling 2D shaders", error);
        return false;
    }

    MTL::Function *vs = library->newFunction(NS::String::string("quad_vs", NS::UTF8StringEncoding));
    MTL::Function *fs = library->newFunction(NS::String::string("quad_fs", NS::UTF8StringEncoding));

    MTL::RenderPipelineDescriptor *desc = MTL::RenderPipelineDescriptor::alloc()->init();
    desc->setLabel(NS::String::string("stretch_pic", NS::UTF8StringEncoding));
    desc->setVertexFunction(vs);
    desc->setFragmentFunction(fs);

    MTL::RenderPipelineColorAttachmentDescriptor *color = desc->colorAttachments()->object(0);
    color->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
    color->setBlendingEnabled(true);
    color->setRgbBlendOperation(MTL::BlendOperationAdd);
    color->setAlphaBlendOperation(MTL::BlendOperationAdd);
    color->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
    color->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    color->setSourceAlphaBlendFactor(MTL::BlendFactorSourceAlpha);
    color->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);

    mtl.quad_pipeline = mtl.device->newRenderPipelineState(desc, &error);

    desc->release();
    vs->release();
    fs->release();
    library->release();

    if (!mtl.quad_pipeline) {
        report_error("creating 2D pipeline", error);
        return false;
    }
    return true;
}

static MTL::SamplerState *create_sampler(mtl_filter_t filter)
{
    MTL::SamplerDescriptor *desc = MTL::SamplerDescriptor::alloc()->init();
    MTL::SamplerMinMagFilter mm = filter == MTL_FILTER_NEAREST
        ? MTL::SamplerMinMagFilterNearest : MTL::SamplerMinMagFilterLinear;
    desc->setMinFilter(mm);
    desc->setMagFilter(mm);
    desc->setMipFilter(MTL::SamplerMipFilterLinear);
    // Repeat, because R_TileClear draws texture coordinates past 1.
    desc->setSAddressMode(MTL::SamplerAddressModeRepeat);
    desc->setTAddressMode(MTL::SamplerAddressModeRepeat);
    MTL::SamplerState *sampler = mtl.device->newSamplerState(desc);
    desc->release();
    return sampler;
}

static void ensure_backbuffer(NS::UInteger width, NS::UInteger height)
{
    if (mtl.backbuffer && mtl.backbuffer->width() == width && mtl.backbuffer->height() == height)
        return;

    if (mtl.backbuffer)
        mtl.backbuffer->release();

    MTL::TextureDescriptor *desc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatBGRA8Unorm_sRGB, width, height, false);
    desc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
    desc->setStorageMode(MTL::StorageModePrivate);
    mtl.backbuffer = mtl.device->newTexture(desc);
    mtl.backbuffer->setLabel(NS::String::string("backbuffer", NS::UTF8StringEncoding));
    mtl.have_presented_frame = false;
}

extern "C" bool mtl_init(void *layer, int width, int height)
{
    memset(&mtl, 0, sizeof(mtl));

    mtl.device = MTL::CreateSystemDefaultDevice();
    if (!mtl.device) {
        Com_LPrintf(MTL_PRINT_ERROR, "Metal: no Metal device available\n");
        return false;
    }

    strncpy(mtl.device_name, mtl.device->name()->utf8String(), sizeof(mtl.device_name) - 1);

    mtl.queue = mtl.device->newCommandQueue();

    mtl.layer = static_cast<CA::MetalLayer *>(layer);
    mtl.layer->setDevice(mtl.device);
    mtl.layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
    // The backbuffer is blitted into the drawable, which needs a non-framebuffer-only texture.
    mtl.layer->setFramebufferOnly(false);
    mtl.layer->setMaximumDrawableCount(3);
    mtl.layer->setDrawableSize(CGSizeMake(width, height));

    if (!create_quad_pipeline())
        return false;

    mtl.samplers[MTL_FILTER_LINEAR] = create_sampler(MTL_FILTER_LINEAR);
    mtl.samplers[MTL_FILTER_NEAREST] = create_sampler(MTL_FILTER_NEAREST);

    for (int i = 0; i < MTL_MAX_FRAMES_IN_FLIGHT; i++) {
        mtl.quad_buffers[i] = mtl.device->newBuffer(MAX_QUADS * sizeof(mtl_quad_t),
                                                    MTL::ResourceStorageModeShared);
    }
    mtl.in_flight = dispatch_semaphore_create(MTL_MAX_FRAMES_IN_FLIGHT);

    MTL::TextureDescriptor *desc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRGBA8Unorm, 1, 1, false);
    desc->setUsage(MTL::TextureUsageShaderRead);
    mtl.white = mtl.device->newTexture(desc);
    const uint32_t white = 0xffffffffu;
    mtl.white->replaceRegion(MTL::Region::Make2D(0, 0, 1, 1), 0, &white, 4);

    Com_LPrintf(MTL_PRINT_ALL, "Metal device: %s (ray tracing %s)\n", mtl.device_name,
                mtl.device->supportsRaytracing() ? "supported" : "not supported");
    return true;
}

static void flush_uploads(void)
{
    if (!mtl.upload_cmd)
        return;
    mtl.upload_blit->endEncoding();
    mtl.upload_cmd->commit();
    mtl.upload_blit->release();
    mtl.upload_cmd->release();
    mtl.upload_blit = nullptr;
    mtl.upload_cmd = nullptr;
}

extern "C" void mtl_wait_idle(void)
{
    if (!mtl.queue)
        return;
    flush_uploads();
    MTL::CommandBuffer *cmd = mtl.queue->commandBuffer();
    cmd->commit();
    cmd->waitUntilCompleted();
}

extern "C" void mtl_shutdown(void)
{
    if (!mtl.device)
        return;

    mtl_wait_idle();
    mtl_texture_free_all();

    for (int i = 0; i < MTL_MAX_FRAMES_IN_FLIGHT; i++) {
        if (mtl.quad_buffers[i])
            mtl.quad_buffers[i]->release();
    }
    if (mtl.backbuffer)
        mtl.backbuffer->release();
    if (mtl.white)
        mtl.white->release();
    for (int i = 0; i < 2; i++) {
        if (mtl.samplers[i])
            mtl.samplers[i]->release();
    }
    if (mtl.quad_pipeline)
        mtl.quad_pipeline->release();
    if (mtl.in_flight)
        dispatch_release(mtl.in_flight);
    mtl.queue->release();
    mtl.device->release();

    memset(&mtl, 0, sizeof(mtl));
}

extern "C" const char *mtl_device_name(void)
{
    return mtl.device_name;
}

extern "C" bool mtl_supports_raytracing(void)
{
    return mtl.device && mtl.device->supportsRaytracing();
}

extern "C" void mtl_set_drawable_size(int width, int height)
{
    if (mtl.layer && width > 0 && height > 0)
        mtl.layer->setDrawableSize(CGSizeMake(width, height));
}

extern "C" void mtl_set_vsync(bool enabled)
{
    if (mtl.layer)
        mtl.layer->setDisplaySyncEnabled(enabled);
}

static MTL::Texture *texture_for_slot(uint32_t slot, mtl_filter_t *filter)
{
    if (slot < MTL_MAX_TEXTURES && mtl.textures[slot].texture) {
        *filter = mtl.textures[slot].filter;
        return mtl.textures[slot].texture;
    }
    *filter = MTL_FILTER_NEAREST;
    return mtl.white;
}

extern "C" bool mtl_texture_upload(uint32_t slot, int width, int height, mtl_format_t format,
                                   const void *pixels, bool srgb, bool mipmaps, mtl_filter_t filter)
{
    if (slot >= MTL_MAX_TEXTURES || width <= 0 || height <= 0 || !pixels)
        return false;

    mtl_texture_free(slot);

    MTL::PixelFormat pf;
    NS::UInteger bpp;
    if (format == MTL_FORMAT_R16) {
        pf = MTL::PixelFormatR16Unorm;
        bpp = 2;
        srgb = false;
    } else {
        pf = srgb ? MTL::PixelFormatRGBA8Unorm_sRGB : MTL::PixelFormatRGBA8Unorm;
        bpp = 4;
    }

    MTL::TextureDescriptor *desc = MTL::TextureDescriptor::texture2DDescriptor(pf, width, height, mipmaps);
    desc->setUsage(MTL::TextureUsageShaderRead);
    MTL::Texture *texture = mtl.device->newTexture(desc);
    if (!texture)
        return false;

    texture->replaceRegion(MTL::Region::Make2D(0, 0, width, height), 0, pixels, width * bpp);

    if (mipmaps && texture->mipmapLevelCount() > 1) {
        if (!mtl.upload_cmd) {
            mtl.upload_cmd = mtl.queue->commandBuffer()->retain();
            mtl.upload_blit = mtl.upload_cmd->blitCommandEncoder()->retain();
        }
        mtl.upload_blit->generateMipmaps(texture);
    }

    mtl.textures[slot].texture = texture;
    mtl.textures[slot].filter = filter;
    return true;
}

extern "C" void mtl_texture_set_filter(uint32_t slot, mtl_filter_t filter)
{
    if (slot < MTL_MAX_TEXTURES)
        mtl.textures[slot].filter = filter;
}

extern "C" void mtl_texture_free(uint32_t slot)
{
    // Command buffers retain the textures they reference, so releasing here is safe
    // even if an in-flight frame still samples this texture.
    if (slot < MTL_MAX_TEXTURES && mtl.textures[slot].texture) {
        mtl.textures[slot].texture->release();
        mtl.textures[slot].texture = nullptr;
    }
}

extern "C" void mtl_texture_free_all(void)
{
    for (uint32_t i = 0; i < MTL_MAX_TEXTURES; i++)
        mtl_texture_free(i);
}

static void encode_quads(MTL::CommandBuffer *cmd, const mtl_quad_t *quads, int num_quads)
{
    MTL::RenderPassDescriptor *pass = MTL::RenderPassDescriptor::renderPassDescriptor();
    MTL::RenderPassColorAttachmentDescriptor *color = pass->colorAttachments()->object(0);
    color->setTexture(mtl.backbuffer);
    color->setLoadAction(MTL::LoadActionClear);
    color->setClearColor(MTL::ClearColor::Make(0.0, 0.0, 0.0, 1.0));
    color->setStoreAction(MTL::StoreActionStore);

    MTL::RenderCommandEncoder *enc = cmd->renderCommandEncoder(pass);
    enc->setLabel(NS::String::string("2D", NS::UTF8StringEncoding));

    if (num_quads > MAX_QUADS)
        num_quads = MAX_QUADS;

    if (num_quads > 0) {
        MTL::Buffer *buffer = mtl.quad_buffers[mtl.frame_index];
        memcpy(buffer->contents(), quads, num_quads * sizeof(mtl_quad_t));

        enc->setRenderPipelineState(mtl.quad_pipeline);
        enc->setVertexBuffer(buffer, 0, 0);

        // Draw runs of consecutive quads that share a texture as one instanced draw,
        // keeping submission order (and so blending order) intact.
        int start = 0;
        while (start < num_quads) {
            int end = start + 1;
            while (end < num_quads && quads[end].tex == quads[start].tex)
                end++;

            mtl_filter_t filter;
            MTL::Texture *texture = texture_for_slot(quads[start].tex, &filter);

            enc->setVertexBufferOffset(start * sizeof(mtl_quad_t), 0);
            enc->setFragmentTexture(texture, 0);
            enc->setFragmentSamplerState(mtl.samplers[filter], 0);
            enc->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4),
                                NS::UInteger(end - start));
            start = end;
        }
    }

    enc->endEncoding();
}

extern "C" bool mtl_end_frame(const mtl_quad_t *quads, int num_quads)
{
    if (!mtl.device)
        return false;

    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

    flush_uploads();

    dispatch_semaphore_wait(mtl.in_flight, DISPATCH_TIME_FOREVER);

    CA::MetalDrawable *drawable = mtl.layer->nextDrawable();
    if (!drawable) {
        dispatch_semaphore_signal(mtl.in_flight);
        pool->release();
        return false;
    }

    MTL::Texture *target = drawable->texture();
    ensure_backbuffer(target->width(), target->height());

    MTL::CommandBuffer *cmd = mtl.queue->commandBuffer();
    cmd->setLabel(NS::String::string("frame", NS::UTF8StringEncoding));

    encode_quads(cmd, quads, num_quads);

    MTL::BlitCommandEncoder *blit = cmd->blitCommandEncoder();
    blit->copyFromTexture(mtl.backbuffer, target);
    blit->endEncoding();

    dispatch_semaphore_t in_flight = mtl.in_flight;
    cmd->addCompletedHandler([in_flight](MTL::CommandBuffer *) {
        dispatch_semaphore_signal(in_flight);
    });
    cmd->presentDrawable(drawable);
    cmd->commit();

    mtl.frame_index = (mtl.frame_index + 1) % MTL_MAX_FRAMES_IN_FLIGHT;
    mtl.have_presented_frame = true;

    pool->release();
    return true;
}

extern "C" bool mtl_read_last_frame_rgb(uint8_t *dst, int width, int height)
{
    if (!mtl.backbuffer || !mtl.have_presented_frame)
        return false;
    if ((NS::UInteger)width != mtl.backbuffer->width() || (NS::UInteger)height != mtl.backbuffer->height())
        return false;

    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

    const NS::UInteger row_bytes = (NS::UInteger)width * 4;
    MTL::Buffer *staging = mtl.device->newBuffer(row_bytes * height, MTL::ResourceStorageModeShared);

    MTL::CommandBuffer *cmd = mtl.queue->commandBuffer();
    MTL::BlitCommandEncoder *blit = cmd->blitCommandEncoder();
    blit->copyFromTexture(mtl.backbuffer, 0, 0, MTL::Origin::Make(0, 0, 0),
                          MTL::Size::Make(width, height, 1), staging, 0, row_bytes, row_bytes * height);
    blit->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    // BGRA top-down -> RGB bottom-up.
    const uint8_t *src = static_cast<const uint8_t *>(staging->contents());
    for (int y = 0; y < height; y++) {
        const uint8_t *in = src + (NS::UInteger)(height - 1 - y) * row_bytes;
        uint8_t *out = dst + (size_t)y * width * 3;
        for (int x = 0; x < width; x++, in += 4, out += 3) {
            out[0] = in[2];
            out[1] = in[1];
            out[2] = in[0];
        }
    }

    staging->release();
    pool->release();
    return true;
}
