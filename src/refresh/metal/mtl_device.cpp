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
// Frame model: everything for a frame is recorded in mtl_end_frame. If a 3D view was
// queued with mtl_render_view, the top-level acceleration structure is rebuilt from the
// frame's instances and a compute pass traces the view into a texture. That texture and
// then the 2D quads are drawn into an offscreen backbuffer which is copied into the
// drawable, so the last presented image can always be read back for screenshots. Per-frame CPU-written buffers
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

// First-hit view: one primary ray per pixel against the scene, shaded with the surface
// texture and a headlight term. This is the stand-in for the path tracer's primary ray
// pass and shares its inputs (TLAS, per-triangle data, bindless textures).
static const char *s_first_hit_shader_source = R"MSL(
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

#define TRI_SKY 1u

struct Triangle {
    packed_float2 uv[3];
    uint tex;
    uint flags;
};

struct View {
    packed_float3 origin;
    float tan_half_fov_x;
    packed_float3 forward;
    float tan_half_fov_y;
    packed_float3 right;
    uint white_texture;
    packed_float3 up;
    uint pad;
};

struct TextureEntry {
    texture2d<float> tex;
};

static float3 sky_color(float3 dir)
{
    return mix(float3(0.35, 0.45, 0.6), float3(0.05, 0.15, 0.4), saturate(dir.z));
}

static float3 view_direction(constant View &view, float2 pixel, float2 size)
{
    float2 ndc = pixel / size * 2.0 - 1.0;
    return normalize(float3(view.forward)
                     + float3(view.right) * (ndc.x * view.tan_half_fov_x)
                     - float3(view.up) * (ndc.y * view.tan_half_fov_y));
}

// Texture coordinates where a ray from origin along dir meets the triangle's plane.
static float2 plane_uv(float3 origin, float3 dir, float3 p0, float3 e1, float3 e2, float3 n,
                       float2 uv0, float2 uv1, float2 uv2)
{
    float t = dot(p0 - origin, n) / dot(dir, n);
    float3 v = origin + dir * t - p0;
    float b1 = dot(cross(v, e2), n) / dot(n, n);
    float b2 = dot(cross(e1, v), n) / dot(n, n);
    return uv0 + (uv1 - uv0) * b1 + (uv2 - uv0) * b2;
}

kernel void first_hit(uint2 tid [[thread_position_in_grid]],
                      texture2d<float, access::write> out [[texture(0)]],
                      instance_acceleration_structure tlas [[buffer(0)]],
                      constant View &view [[buffer(1)]],
                      const device Triangle *triangles [[buffer(2)]],
                      const device packed_float3 *positions [[buffer(3)]],
                      const device TextureEntry *textures [[buffer(4)]],
                      sampler smp [[sampler(0)]])
{
    uint2 size = uint2(out.get_width(), out.get_height());
    if (tid.x >= size.x || tid.y >= size.y)
        return;

    ray r;
    r.origin = view.origin;
    r.direction = view_direction(view, float2(tid) + 0.5, float2(size));
    r.min_distance = 0.0;
    r.max_distance = INFINITY;

    intersection_params params;
    params.assume_geometry_type(geometry_type::triangle);
    params.force_opacity(forced_opacity::opaque);

    intersection_query<triangle_data, instancing> query(r, tlas, 0xff, params);
    while (query.next()) {}

    if (query.get_committed_intersection_type() != intersection_type::triangle) {
        out.write(float4(0.0, 0.0, 0.0, 1.0), tid);
        return;
    }

    // The instance's user ID is the index of its model's first triangle.
    uint index = query.get_committed_user_instance_id() + query.get_committed_primitive_id();
    Triangle tri = triangles[index];

    if (tri.flags & TRI_SKY) {
        out.write(float4(sky_color(r.direction), 1.0), tid);
        return;
    }

    float4x3 object_to_world = query.get_committed_object_to_world_transform();
    float3 p0 = object_to_world * float4(float3(positions[index * 3 + 0]), 1.0);
    float3 e1 = object_to_world * float4(float3(positions[index * 3 + 1]), 1.0) - p0;
    float3 e2 = object_to_world * float4(float3(positions[index * 3 + 2]), 1.0) - p0;
    float3 n = cross(e1, e2);

    // Texture gradients from the rays through the neighbouring pixels.
    float2 uv0 = tri.uv[0], uv1 = tri.uv[1], uv2 = tri.uv[2];
    float2 pixel = float2(tid) + 0.5;
    float2 uv = plane_uv(r.origin, r.direction, p0, e1, e2, n, uv0, uv1, uv2);
    float2 uv_x = plane_uv(r.origin, view_direction(view, pixel + float2(1, 0), float2(size)),
                           p0, e1, e2, n, uv0, uv1, uv2);
    float2 uv_y = plane_uv(r.origin, view_direction(view, pixel + float2(0, 1), float2(size)),
                           p0, e1, e2, n, uv0, uv1, uv2);

    float3 albedo = textures[min(tri.tex, view.white_texture)].tex.sample(
        smp, uv, gradient2d(uv_x - uv, uv_y - uv)).rgb;
    float headlight = abs(dot(normalize(n), r.direction));

    out.write(float4(albedo * (0.25 + 0.75 * headlight), 1.0), tid);
}
)MSL";

// Matches View in the first-hit shader.
struct view_uniforms_t {
    float origin[3];
    float tan_half_fov_x;
    float forward[3];
    float tan_half_fov_y;
    float right[3];
    uint32_t white_texture;
    float up[3];
    uint32_t pad;
};

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

    int width, height;
    bool offscreen;

    // World geometry: unindexed triangles and one BLAS per model that has any.
    MTL::Buffer *positions;
    MTL::Buffer *triangles;
    mtl_model_t *models;
    uint32_t *model_blas;  // index into blas_array, or UINT32_MAX for an empty model
    int num_models;
    NS::Array *blas_array;

    // The TLAS is sized for MTL_MAX_INSTANCES and rebuilt every frame. Frames execute in
    // submission order, so one TLAS and one scratch buffer serve all frames in flight.
    MTL::AccelerationStructure *tlas;
    MTL::Buffer *tlas_scratch;
    MTL::Buffer *instance_buffers[MTL_MAX_FRAMES_IN_FLIGHT];

    // Bindless texture table: slot -> texture, with the white texture at MTL_MAX_TEXTURES.
    // Rewritten for a frame only if a texture changed since that buffer was last filled.
    MTL::Buffer *texture_tables[MTL_MAX_FRAMES_IN_FLIGHT];
    uint64_t texture_table_generation[MTL_MAX_FRAMES_IN_FLIGHT];
    uint64_t texture_generation;

    MTL::ComputePipelineState *first_hit_pipeline;
    MTL::SamplerState *world_sampler;
    MTL::Texture *view_texture;

    bool have_view;
    mtl_view_t view;
    mtl_instance_t view_instances[MTL_MAX_INSTANCES];

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

static bool create_first_hit_pipeline(void)
{
    NS::Error *error = nullptr;
    NS::String *source = NS::String::string(s_first_hit_shader_source, NS::UTF8StringEncoding);
    MTL::Library *library = mtl.device->newLibrary(source, nullptr, &error);
    if (!library) {
        report_error("compiling first-hit shader", error);
        return false;
    }

    MTL::Function *fn = library->newFunction(NS::String::string("first_hit", NS::UTF8StringEncoding));
    mtl.first_hit_pipeline = mtl.device->newComputePipelineState(fn, &error);

    fn->release();
    library->release();

    if (!mtl.first_hit_pipeline) {
        report_error("creating first-hit pipeline", error);
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
    if (mtl.view_texture)
        mtl.view_texture->release();

    MTL::TextureDescriptor *view_desc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRGBA16Float, width, height, false);
    view_desc->setUsage(MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead);
    view_desc->setStorageMode(MTL::StorageModePrivate);
    mtl.view_texture = mtl.device->newTexture(view_desc);
    mtl.view_texture->setLabel(NS::String::string("view", NS::UTF8StringEncoding));

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
    mtl.width = width;
    mtl.height = height;

    if (!create_quad_pipeline())
        return false;

    if (mtl.device->supportsRaytracing()) {
        if (!create_first_hit_pipeline())
            return false;

        MTL::SamplerDescriptor *sampler_desc = MTL::SamplerDescriptor::alloc()->init();
        sampler_desc->setMinFilter(MTL::SamplerMinMagFilterLinear);
        sampler_desc->setMagFilter(MTL::SamplerMinMagFilterLinear);
        sampler_desc->setMipFilter(MTL::SamplerMipFilterLinear);
        sampler_desc->setSAddressMode(MTL::SamplerAddressModeRepeat);
        sampler_desc->setTAddressMode(MTL::SamplerAddressModeRepeat);
        sampler_desc->setMaxAnisotropy(8);
        mtl.world_sampler = mtl.device->newSamplerState(sampler_desc);
        sampler_desc->release();

        for (int i = 0; i < MTL_MAX_FRAMES_IN_FLIGHT; i++) {
            mtl.instance_buffers[i] = mtl.device->newBuffer(
                MTL_MAX_INSTANCES * sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor),
                MTL::ResourceStorageModeShared);
            mtl.texture_tables[i] = mtl.device->newBuffer(
                (MTL_MAX_TEXTURES + 1) * sizeof(MTL::ResourceID), MTL::ResourceStorageModeShared);
        }
        mtl.texture_generation = 1;
    }

    mtl.samplers[MTL_FILTER_LINEAR] = create_sampler(MTL_FILTER_LINEAR);
    mtl.samplers[MTL_FILTER_NEAREST] = create_sampler(MTL_FILTER_NEAREST);

    for (int i = 0; i < MTL_MAX_FRAMES_IN_FLIGHT; i++) {
        // one extra quad at the end for the 3D view
        mtl.quad_buffers[i] = mtl.device->newBuffer((MAX_QUADS + 1) * sizeof(mtl_quad_t),
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

    mtl_world_free();
    mtl_texture_free_all();

    for (int i = 0; i < MTL_MAX_FRAMES_IN_FLIGHT; i++) {
        if (mtl.quad_buffers[i])
            mtl.quad_buffers[i]->release();
        if (mtl.instance_buffers[i])
            mtl.instance_buffers[i]->release();
        if (mtl.texture_tables[i])
            mtl.texture_tables[i]->release();
    }
    if (mtl.first_hit_pipeline)
        mtl.first_hit_pipeline->release();
    if (mtl.world_sampler)
        mtl.world_sampler->release();
    if (mtl.view_texture)
        mtl.view_texture->release();
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
    if (mtl.layer && width > 0 && height > 0) {
        mtl.layer->setDrawableSize(CGSizeMake(width, height));
        mtl.width = width;
        mtl.height = height;
    }
}

extern "C" void mtl_set_offscreen(bool enabled)
{
    mtl.offscreen = enabled;
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
    mtl.texture_generation++;
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
        mtl.texture_generation++;
    }
}

extern "C" void mtl_texture_free_all(void)
{
    for (uint32_t i = 0; i < MTL_MAX_TEXTURES; i++)
        mtl_texture_free(i);
}

extern "C" void mtl_world_free(void)
{
    if (!mtl.positions)
        return;

    mtl_wait_idle();

    mtl.have_view = false;
    mtl.positions->release();
    mtl.triangles->release();
    mtl.blas_array->release();
    mtl.tlas->release();
    mtl.tlas_scratch->release();
    delete[] mtl.models;
    delete[] mtl.model_blas;

    mtl.positions = nullptr;
    mtl.triangles = nullptr;
    mtl.blas_array = nullptr;
    mtl.tlas = nullptr;
    mtl.tlas_scratch = nullptr;
    mtl.models = nullptr;
    mtl.model_blas = nullptr;
    mtl.num_models = 0;
}

static MTL::InstanceAccelerationStructureDescriptor *tlas_descriptor(MTL::Buffer *instances,
                                                                     NS::UInteger count)
{
    MTL::InstanceAccelerationStructureDescriptor *desc =
        MTL::InstanceAccelerationStructureDescriptor::descriptor();
    desc->setInstancedAccelerationStructures(mtl.blas_array);
    desc->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeUserID);
    desc->setInstanceDescriptorBuffer(instances);
    desc->setInstanceCount(count);
    return desc;
}

extern "C" bool mtl_world_upload(const float *positions, const mtl_triangle_t *triangles,
                                 int num_triangles, const mtl_model_t *models, int num_models)
{
    mtl_world_free();

    if (!mtl.first_hit_pipeline || num_triangles <= 0 || num_models <= 0)
        return false;

    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

    const NS::UInteger vertex_stride = 3 * sizeof(float);

    mtl.positions = mtl.device->newBuffer(positions, (NS::UInteger)num_triangles * 3 * vertex_stride,
                                          MTL::ResourceStorageModeShared);
    mtl.triangles = mtl.device->newBuffer(triangles, (NS::UInteger)num_triangles * sizeof(mtl_triangle_t),
                                          MTL::ResourceStorageModeShared);
    mtl.positions->setLabel(NS::String::string("world positions", NS::UTF8StringEncoding));
    mtl.triangles->setLabel(NS::String::string("world triangles", NS::UTF8StringEncoding));

    mtl.models = new mtl_model_t[num_models];
    mtl.model_blas = new uint32_t[num_models];
    mtl.num_models = num_models;
    memcpy(mtl.models, models, num_models * sizeof(mtl_model_t));

    MTL::CommandBuffer *cmd = mtl.queue->commandBuffer();
    MTL::AccelerationStructureCommandEncoder *enc = cmd->accelerationStructureCommandEncoder();

    MTL::AccelerationStructure **blas = new MTL::AccelerationStructure *[num_models];
    uint32_t num_blas = 0;

    for (int i = 0; i < num_models; i++) {
        mtl.model_blas[i] = UINT32_MAX;
        if (!models[i].num_triangles)
            continue;

        MTL::AccelerationStructureTriangleGeometryDescriptor *geometry =
            MTL::AccelerationStructureTriangleGeometryDescriptor::descriptor();
        geometry->setVertexBuffer(mtl.positions);
        geometry->setVertexBufferOffset((NS::UInteger)models[i].first_triangle * 3 * vertex_stride);
        geometry->setVertexStride(vertex_stride);
        geometry->setVertexFormat(MTL::AttributeFormatFloat3);
        geometry->setTriangleCount(models[i].num_triangles);
        geometry->setOpaque(true);

        MTL::PrimitiveAccelerationStructureDescriptor *desc =
            MTL::PrimitiveAccelerationStructureDescriptor::descriptor();
        desc->setGeometryDescriptors(NS::Array::array(geometry));

        MTL::AccelerationStructureSizes sizes = mtl.device->accelerationStructureSizes(desc);
        MTL::AccelerationStructure *as = mtl.device->newAccelerationStructure(sizes.accelerationStructureSize);
        MTL::Buffer *scratch = mtl.device->newBuffer(sizes.buildScratchBufferSize,
                                                     MTL::ResourceStorageModePrivate);
        enc->buildAccelerationStructure(as, desc, scratch, 0);
        scratch->release();  // kept alive by the command buffer

        mtl.model_blas[i] = num_blas;
        blas[num_blas++] = as;
    }

    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    mtl.blas_array = NS::Array::array(reinterpret_cast<const NS::Object *const *>(blas), num_blas)->retain();
    for (uint32_t i = 0; i < num_blas; i++)
        blas[i]->release();  // now owned by the array
    delete[] blas;

    MTL::AccelerationStructureSizes sizes = mtl.device->accelerationStructureSizes(
        tlas_descriptor(mtl.instance_buffers[0], MTL_MAX_INSTANCES));
    mtl.tlas = mtl.device->newAccelerationStructure(sizes.accelerationStructureSize);
    mtl.tlas->setLabel(NS::String::string("TLAS", NS::UTF8StringEncoding));
    mtl.tlas_scratch = mtl.device->newBuffer(sizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);

    bool ok = cmd->status() == MTL::CommandBufferStatusCompleted && num_blas > 0;
    if (cmd->status() != MTL::CommandBufferStatusCompleted)
        report_error("building acceleration structures", cmd->error());

    pool->release();

    if (!ok)
        mtl_world_free();
    return ok;
}

extern "C" void mtl_render_view(const mtl_view_t *view)
{
    if (!mtl.tlas)
        return;

    int count = view->num_instances < MTL_MAX_INSTANCES ? view->num_instances : MTL_MAX_INSTANCES;
    memcpy(mtl.view_instances, view->instances, count * sizeof(mtl_instance_t));

    mtl.view = *view;
    mtl.view.instances = mtl.view_instances;
    mtl.view.num_instances = count;
    mtl.have_view = true;
}

// Rebuilds the TLAS from the queued view's instances and traces the view into
// mtl.view_texture. Returns false if there was nothing to trace.
static bool encode_view(MTL::CommandBuffer *cmd)
{
    if (!mtl.have_view || !mtl.tlas)
        return false;

    MTL::Buffer *instance_buffer = mtl.instance_buffers[mtl.frame_index];
    auto *instances = static_cast<MTL::AccelerationStructureUserIDInstanceDescriptor *>(instance_buffer->contents());
    NS::UInteger count = 0;

    for (int i = 0; i < mtl.view.num_instances; i++) {
        const mtl_instance_t *in = &mtl.view.instances[i];
        if (in->model >= (uint32_t)mtl.num_models || mtl.model_blas[in->model] == UINT32_MAX)
            continue;

        MTL::AccelerationStructureUserIDInstanceDescriptor *out = &instances[count++];
        memset(out, 0, sizeof(*out));
        for (int c = 0; c < 3; c++)
            out->transformationMatrix.columns[c] = MTL::PackedFloat3(in->axis[c][0], in->axis[c][1], in->axis[c][2]);
        out->transformationMatrix.columns[3] = MTL::PackedFloat3(in->origin[0], in->origin[1], in->origin[2]);
        out->options = MTL::AccelerationStructureInstanceOptionOpaque;
        out->mask = 0xff;
        out->accelerationStructureIndex = mtl.model_blas[in->model];
        out->userID = mtl.models[in->model].first_triangle;
    }

    if (!count)
        return false;

    MTL::AccelerationStructureCommandEncoder *as_enc = cmd->accelerationStructureCommandEncoder();
    as_enc->buildAccelerationStructure(mtl.tlas, tlas_descriptor(instance_buffer, count), mtl.tlas_scratch, 0);
    as_enc->endEncoding();

    // Textures reached through the table are not visible to Metal as bindings, so each
    // one has to be declared to the encoder.
    static const MTL::Resource *used[MTL_MAX_TEXTURES + 1];
    NS::UInteger num_used = 0;

    MTL::Buffer *table = mtl.texture_tables[mtl.frame_index];
    bool refill = mtl.texture_table_generation[mtl.frame_index] != mtl.texture_generation;
    auto *ids = static_cast<MTL::ResourceID *>(table->contents());

    for (uint32_t i = 0; i < MTL_MAX_TEXTURES; i++) {
        MTL::Texture *texture = mtl.textures[i].texture;
        if (texture)
            used[num_used++] = texture;
        if (refill)
            ids[i] = (texture ? texture : mtl.white)->gpuResourceID();
    }
    used[num_used++] = mtl.white;
    if (refill) {
        ids[MTL_MAX_TEXTURES] = mtl.white->gpuResourceID();
        mtl.texture_table_generation[mtl.frame_index] = mtl.texture_generation;
    }

    view_uniforms_t uniforms = {};
    memcpy(uniforms.origin, mtl.view.origin, sizeof(uniforms.origin));
    memcpy(uniforms.forward, mtl.view.forward, sizeof(uniforms.forward));
    memcpy(uniforms.right, mtl.view.right, sizeof(uniforms.right));
    memcpy(uniforms.up, mtl.view.up, sizeof(uniforms.up));
    uniforms.tan_half_fov_x = mtl.view.tan_half_fov_x;
    uniforms.tan_half_fov_y = mtl.view.tan_half_fov_y;
    uniforms.white_texture = MTL_MAX_TEXTURES;

    MTL::ComputeCommandEncoder *enc = cmd->computeCommandEncoder();
    enc->setLabel(NS::String::string("first hit", NS::UTF8StringEncoding));
    enc->setComputePipelineState(mtl.first_hit_pipeline);
    enc->setTexture(mtl.view_texture, 0);
    enc->setAccelerationStructure(mtl.tlas, 0);
    enc->setBytes(&uniforms, sizeof(uniforms), 1);
    enc->setBuffer(mtl.triangles, 0, 2);
    enc->setBuffer(mtl.positions, 0, 3);
    enc->setBuffer(table, 0, 4);
    enc->setSamplerState(mtl.world_sampler, 0);
    enc->useResources(used, num_used, MTL::ResourceUsageRead);
    for (NS::UInteger i = 0; i < mtl.blas_array->count(); i++)
        enc->useResource(mtl.blas_array->object<MTL::AccelerationStructure>(i), MTL::ResourceUsageRead);

    const NS::UInteger group = 8;
    NS::UInteger width = mtl.view_texture->width();
    NS::UInteger height = mtl.view_texture->height();
    enc->dispatchThreadgroups(MTL::Size::Make((width + group - 1) / group, (height + group - 1) / group, 1),
                              MTL::Size::Make(group, group, 1));
    enc->endEncoding();
    return true;
}

static void encode_quads(MTL::CommandBuffer *cmd, const mtl_quad_t *quads, int num_quads, bool draw_view)
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

    MTL::Buffer *buffer = mtl.quad_buffers[mtl.frame_index];

    if (draw_view || num_quads > 0) {
        enc->setRenderPipelineState(mtl.quad_pipeline);
        enc->setVertexBuffer(buffer, 0, 0);
    }

    if (draw_view) {
        mtl_quad_t *quad = static_cast<mtl_quad_t *>(buffer->contents()) + MAX_QUADS;
        *quad = { -1.0f, -1.0f, 2.0f, 2.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0xffffffffu, MTL_TEX_WHITE };

        enc->setVertexBufferOffset(MAX_QUADS * sizeof(mtl_quad_t), 0);
        enc->setFragmentTexture(mtl.view_texture, 0);
        enc->setFragmentSamplerState(mtl.samplers[MTL_FILTER_NEAREST], 0);
        enc->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4), NS::UInteger(1));
    }

    if (num_quads > 0) {
        memcpy(buffer->contents(), quads, num_quads * sizeof(mtl_quad_t));

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

    CA::MetalDrawable *drawable = nullptr;
    if (mtl.offscreen) {
        ensure_backbuffer(mtl.width, mtl.height);
    } else {
        drawable = mtl.layer->nextDrawable();
        if (!drawable) {
            mtl.have_view = false;
            dispatch_semaphore_signal(mtl.in_flight);
            pool->release();
            return false;
        }
        ensure_backbuffer(drawable->texture()->width(), drawable->texture()->height());
    }

    MTL::CommandBuffer *cmd = mtl.queue->commandBuffer();
    cmd->setLabel(NS::String::string("frame", NS::UTF8StringEncoding));

    bool draw_view = encode_view(cmd);
    mtl.have_view = false;

    encode_quads(cmd, quads, num_quads, draw_view);

    if (drawable) {
        MTL::BlitCommandEncoder *blit = cmd->blitCommandEncoder();
        blit->copyFromTexture(mtl.backbuffer, drawable->texture());
        blit->endEncoding();
    }

    dispatch_semaphore_t in_flight = mtl.in_flight;
    cmd->addCompletedHandler([in_flight](MTL::CommandBuffer *) {
        dispatch_semaphore_signal(in_flight);
    });
    if (drawable)
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
