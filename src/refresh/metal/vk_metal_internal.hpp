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

// vk_metal_internal.hpp -- objects behind the Vulkan handles, shared by the vk_metal_*.cpp files.
//
// Every Vulkan handle is a pointer to one of these structs. Only what vkpt uses is
// implemented; anything else is reported through vkmtl_unsupported.

#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <vulkan/vulkan.h>

#include <map>
#include <stdlib.h>
#include <string>
#include <vector>

extern "C" void Com_LPrintf(int type, const char *fmt, ...);
extern "C" int Cvar_VariableInteger(const char *var_name);
#define VKMTL_PRINT_ALL     0
#define VKMTL_PRINT_WARNING 3
#define VKMTL_PRINT_ERROR   4

#define vkmtl_error(...)   Com_LPrintf(VKMTL_PRINT_ERROR, "Metal: " __VA_ARGS__)
#define vkmtl_warning(...) Com_LPrintf(VKMTL_PRINT_WARNING, "Metal: " __VA_ARGS__)
#define vkmtl_print(...)   Com_LPrintf(VKMTL_PRINT_ALL, __VA_ARGS__)

namespace vkmtl {

// Buffer indices in the translated shaders; these match mslgen.
enum {
    MAX_SETS = 8,                    // argument buffer of set N is buffer N
    SAMPLER_SET_OFFSET = 4,          // set N + 4 holds the samplers of set N's texture array
    PUSH_CONSTANT_BUFFER_INDEX = 8,
    VERTEX_BUFFER_BASE_INDEX = 16,   // vertex binding N is buffer 16 + N
    MAX_VERTEX_BINDINGS = 4,
};

inline NS::String *nsstr(const char *s) { return NS::String::string(s, NS::UTF8StringEncoding); }

struct Device;
extern Device g;

// ---------------------------------------------------------------- memory, buffers, images

struct Image;

// Memory is only backed by a real MTLBuffer once something needs its contents or its
// GPU address; memory that images or acceleration structures are "bound" to never is.
struct Memory {
    size_t size;
    MTL::Buffer *buffer;
    uint64_t address;      // start of this memory's range of fake device addresses
    Image *image;          // image bound to this memory, for mapping linear images
    void *image_mapping;

    MTL::Buffer *get();
};

struct Buffer {
    size_t size;
    VkBufferUsageFlags usage;
    Memory *memory;
    size_t offset;

    MTL::Buffer *mtl() const { return memory->get(); }
    uint64_t gpu_address() const { return memory->get()->gpuAddress() + offset; }
};

struct BufferView {
    Buffer *buffer;
    MTL::Texture *texture;
};

struct Image {
    VkImageCreateInfo info;
    MTL::Texture *texture;
    bool is_swapchain;
    std::string name;
};

struct ImageView {
    Image *image;
    MTL::Texture *texture;
    VkImageSubresourceRange range;
};

struct Sampler {
    MTL::SamplerState *state;
};

struct Accel {
    MTL::AccelerationStructure *as;
    VkAccelerationStructureTypeKHR type;
};

MTL::PixelFormat pixel_format(VkFormat format);
uint32_t format_bytes_per_pixel(VkFormat format);
bool resolve_address(uint64_t address, MTL::Buffer **buffer, size_t *offset);

// Everything shaders reach through argument buffers has to be resident.
void residency_add(const MTL::Allocation *allocation);
void residency_remove(const MTL::Allocation *allocation);
void residency_commit(void);

// ---------------------------------------------------------------- descriptors, shaders

struct SetLayoutVk {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    const VkDescriptorSetLayoutBinding *find(uint32_t binding) const;
};

// One descriptor. vkpt may destroy an image view or buffer while a descriptor still
// names it, as Vulkan allows for descriptors that are not used again. The slot therefore
// keeps what goes into the argument buffer by value, and holds a reference on the Metal
// object so that declaring it to an encoder stays valid.
struct Slot {
    VkDescriptorType type;
    uint64_t value;          // resource ID of the texture or acceleration structure, or buffer address
    uint64_t sampler_value;  // resource ID of the sampler
    MTL::Resource *resource;
};

// Where the resources of one descriptor set live in a shader stage's argument buffer.
// Shaders that declare the same resources share one of these.
struct ArgLayout {
    struct Entry {
        uint32_t binding;
        uint32_t count;      // elements declared in the shader; 0 for a runtime array
        int offset;          // of the buffer, texture or acceleration structure, or -1
        int sampler_offset;  // of the sampler of a combined image sampler, or -1
    };
    std::vector<Entry> entries;
    size_t fixed_size;
    bool samplers_only;      // companion set: holds the samplers of the source set's bindings
};

struct DescSet {
    SetLayoutVk *layout;
    std::map<uint32_t, std::vector<Slot>> slots;
    uint64_t version;

    // Metal does not see what a shader reaches through an argument buffer, so it cannot
    // order passes by it, and it does run passes out of order or side by side when it
    // believes they are independent. What a set makes writable (storage buffers and
    // images) and what it only reads (sampled images, uniform and texel buffers,
    // acceleration structures) are listed here and declared to the encoder. Without
    // that an acceleration structure build can run before the compute pass that writes
    // its vertices, and a pass can sample an image before the pass that renders it.
    // The big array of game textures is left out: passes never write those.
    struct Encoded {
        MTL::Buffer *buffer;
        uint64_t version;
        std::vector<const MTL::Resource *> written;
        std::vector<const MTL::Resource *> read;
    };
    std::map<const ArgLayout *, Encoded> encoded;

    const Encoded &encode(const ArgLayout *layout);
};

struct ShaderModule {
    struct Bind {
        uint32_t set, binding, count;
        int id, sampler_id;
        bool runtime;
    };
    std::string source;
    std::string name;
    std::string entry;
    std::string stage;
    bool has_push;
    uint32_t local_size[3];
    std::vector<Bind> binds;
    MTL::Library *library;

    MTL::Function *function(const VkSpecializationInfo *spec);
};

// One shader stage of a pipeline: which argument buffers it takes and how they are laid out.
struct Stage {
    ShaderModule *module;
    const ArgLayout *sets[MAX_SETS];
    bool has_push;

    void reflect(NS::Array *bindings);
};

struct RenderPass {
    std::vector<VkAttachmentDescription> attachments;
    std::vector<VkAttachmentReference> color;
    bool has_depth;
    VkAttachmentReference depth;
};

struct Framebuffer {
    std::vector<ImageView *> views;
    uint32_t width, height;
};

struct Pipeline {
    VkPipelineBindPoint bind_point;
    MTL::ComputePipelineState *compute;
    MTL::RenderPipelineState *render;
    MTL::DepthStencilState *depth_stencil;
    Stage stages[2];  // compute, or vertex and fragment
    MTL::Size threads_per_group;
    MTL::PrimitiveType primitive;
    MTL::CullMode cull;
    MTL::Winding winding;
    MTL::TriangleFillMode fill;
    bool has_viewport, has_scissor;
    VkViewport viewport;
    VkRect2D scissor;
};

// ---------------------------------------------------------------- commands

struct Fence {
    bool signaled;
    MTL::CommandBuffer *pending;
};

struct CommandBuffer {
    MTL::CommandBuffer *cmd;
    MTL::ComputeCommandEncoder *compute;
    MTL::BlitCommandEncoder *blit;
    MTL::RenderCommandEncoder *render;

    Pipeline *pipelines[2];             // by bind point: 0 graphics, 1 compute
    DescSet *sets[2][MAX_SETS];
    uint8_t push[256];
    VkViewport viewport;
    VkRect2D scissor;
    Buffer *index_buffer;
    size_t index_offset;
    VkIndexType index_type;
    Buffer *vertex_buffers[MAX_VERTEX_BINDINGS];
    size_t vertex_offsets[MAX_VERTEX_BINDINGS];
    bool has_viewport, has_scissor;
    uint32_t target_width, target_height;

    void end_encoder();
    MTL::ComputeCommandEncoder *compute_encoder();
    MTL::BlitCommandEncoder *blit_encoder();
};

// Timestamp queries: one counter sample per query.
struct QueryPool {
    MTL::CounterSampleBuffer *samples;
    uint32_t count;
};

struct Surface {
    CA::MetalLayer *layer;
    void *view;  // SDL_MetalView
};

struct Swapchain {
    Surface *surface;
    Image image;
    VkExtent2D extent;
};

struct Device {
    MTL::Device *device;
    MTL::CommandQueue *queue;
    MTL::ResidencySet *residency;
    bool residency_dirty;
    bool offscreen;
    MTL::CommandBuffer *last_submitted;
    std::map<uint64_t, Memory *> memories;  // by fake device address
    uint64_t next_address;
    std::vector<ArgLayout *> arg_layouts;
    std::vector<Image *> images;

    // Places timestamp samples among the passes: every pass reads this buffer and a
    // timestamp marker that has to be exact writes it, so the marker runs after the
    // passes recorded before it and before those recorded after it.
    MTL::Buffer *order_token;
};

// Development aid: with VKMTL_DUMP_FRAME=N in the environment, prints statistics of
// every named 2D image when frame N is presented.
void debug_frame(void);

// Development aid: with VKMTL_PROFILE in the environment every compute dispatch gets a
// command buffer of its own, and the GPU time per shader is printed every 60 frames.
// Committing early changes the order in which vkpt's command buffers run, so the image
// is not reliable in this mode (exposure in particular); the timings are.
bool profiling(void);
void profile_dispatch(CommandBuffer *cb, const char *name);
void profile_report(void);

void submitted(MTL::CommandBuffer *cmd);

void blit_scaled(CommandBuffer *cb, MTL::Texture *src, uint32_t src_level, uint32_t src_slice,
                 const VkOffset3D src_offsets[2], MTL::Texture *dst, uint32_t dst_level,
                 uint32_t dst_slice, const VkOffset3D dst_offsets[2], bool linear);

void vkmtl_unsupported(const char *what);


}  // namespace vkmtl

#define VKMTL_HANDLE(type, handle) (reinterpret_cast<vkmtl::type *>(handle))
