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

// vk_metal_device.cpp -- instance, device, memory, buffers, images, samplers, the
// swapchain and synchronization of the Vulkan subset that vkpt runs on.
//
// There is one device and one command queue. Command buffers are encoded into Metal
// command buffers as they are recorded and committed by vkQueueSubmit, so work runs in
// submission order and semaphores have nothing to do. The swapchain has a single image,
// an ordinary texture that vkQueuePresentKHR copies into the next drawable; that keeps
// the last frame readable for screenshots and lets vid_hidden run without presenting.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "vk_metal_internal.hpp"
#include "vk_metal.h"

#include <SDL2/SDL_metal.h>
#include <QuartzCore/CABase.h>
#include <atomic>
#include <stdlib.h>
#include <string.h>

namespace vkmtl {

Device g;

static NS::AutoreleasePool *frame_pool;
static SDL_Window *surface_window;

// Objects that Metal hands out autoreleased are either retained by this layer or only
// used on the spot, so the pool can be emptied once a frame.
static void drain_pool(void)
{
    if (frame_pool)
        frame_pool->release();
    frame_pool = NS::AutoreleasePool::alloc()->init();
}

void vkmtl_unsupported(const char *what)
{
    vkmtl_error("%s is not implemented\n", what);
}

// ------------------------------------------------------------------------------ formats

MTL::PixelFormat pixel_format(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:      return MTL::PixelFormatRGBA8Unorm;
    case VK_FORMAT_R8G8B8A8_SRGB:       return MTL::PixelFormatRGBA8Unorm_sRGB;
    case VK_FORMAT_B8G8R8A8_UNORM:      return MTL::PixelFormatBGRA8Unorm;
    case VK_FORMAT_B8G8R8A8_SRGB:       return MTL::PixelFormatBGRA8Unorm_sRGB;
    case VK_FORMAT_R8G8_UNORM:          return MTL::PixelFormatRG8Unorm;
    case VK_FORMAT_R16_UNORM:           return MTL::PixelFormatR16Unorm;
    case VK_FORMAT_R16_UINT:            return MTL::PixelFormatR16Uint;
    case VK_FORMAT_R16_SFLOAT:          return MTL::PixelFormatR16Float;
    case VK_FORMAT_R16G16_SFLOAT:       return MTL::PixelFormatRG16Float;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return MTL::PixelFormatRGBA16Float;
    case VK_FORMAT_R32_UINT:            return MTL::PixelFormatR32Uint;
    case VK_FORMAT_R32_SFLOAT:          return MTL::PixelFormatR32Float;
    case VK_FORMAT_R32G32_UINT:         return MTL::PixelFormatRG32Uint;
    case VK_FORMAT_R32G32B32A32_UINT:   return MTL::PixelFormatRGBA32Uint;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return MTL::PixelFormatRGBA32Float;
    case VK_FORMAT_D32_SFLOAT:          return MTL::PixelFormatDepth32Float;
    default:
        vkmtl_error("unsupported image format %d\n", (int)format);
        return MTL::PixelFormatInvalid;
    }
}

uint32_t format_bytes_per_pixel(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SFLOAT:
        return 2;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32_UINT:
        return 8;
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return 16;
    default:
        return 4;
    }
}

static bool is_rgba8(VkFormat format)
{
    return format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_R8G8B8A8_SRGB
        || format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
}

// ------------------------------------------------------------------------------ residency

void residency_add(const MTL::Allocation *allocation)
{
    g.residency->addAllocation(allocation);
    g.residency_dirty = true;
}

void residency_remove(const MTL::Allocation *allocation)
{
    g.residency->removeAllocation(allocation);
    g.residency_dirty = true;
}

void residency_commit(void)
{
    if (g.residency_dirty) {
        g.residency->commit();
        g.residency_dirty = false;
    }
}

// ------------------------------------------------------------------------------ memory

MTL::Buffer *Memory::get()
{
    if (!buffer) {
        buffer = g.device->newBuffer(size, MTL::ResourceStorageModeShared);
        if (!buffer) {
            vkmtl_error("couldn't allocate a %zu byte buffer\n", size);
            abort();
        }
        residency_add(buffer);
    }
    return buffer;
}

bool resolve_address(uint64_t address, MTL::Buffer **buffer, size_t *offset)
{
    auto it = g.memories.upper_bound(address);
    if (it == g.memories.begin())
        return false;
    --it;
    Memory *memory = it->second;
    if (address - memory->address >= memory->size)
        return false;
    *buffer = memory->get();
    *offset = address - memory->address;
    return true;
}

static void wait_idle(void)
{
    if (g.last_submitted) {
        g.last_submitted->waitUntilCompleted();
        g.last_submitted->release();
        g.last_submitted = nullptr;
    }
}

}  // namespace vkmtl

using namespace vkmtl;

extern "C" {

// ------------------------------------------------------------------------------ instance

static int the_instance, the_physical_device, the_device, the_queue;

VkResult vkEnumerateInstanceLayerProperties(uint32_t *count, VkLayerProperties *)
{
    *count = 0;
    return VK_SUCCESS;
}

static VkResult list_extensions(const char *const *names, uint32_t num_names, uint32_t *count,
                                VkExtensionProperties *props)
{
    if (!props) {
        *count = num_names;
        return VK_SUCCESS;
    }
    uint32_t n = *count < num_names ? *count : num_names;
    for (uint32_t i = 0; i < n; i++) {
        memset(&props[i], 0, sizeof(props[i]));
        strncpy(props[i].extensionName, names[i], VK_MAX_EXTENSION_NAME_SIZE - 1);
        props[i].specVersion = 1;
    }
    *count = n;
    return VK_SUCCESS;
}

VkResult vkEnumerateInstanceExtensionProperties(const char *, uint32_t *count, VkExtensionProperties *props)
{
    static const char *const names[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };
    return list_extensions(names, 2, count, props);
}

VkResult vkEnumerateDeviceExtensionProperties(VkPhysicalDevice, const char *, uint32_t *count,
                                              VkExtensionProperties *props)
{
    // Ray queries only: Metal has no counterpart of the ray tracing pipeline's shader
    // binding table that the translated shaders could use.
    static const char *const names[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_BIND_MEMORY_2_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_EXT_DEBUG_MARKER_EXTENSION_NAME,
    };
    return list_extensions(names, 6, count, props);
}

VkResult vkCreateInstance(const VkInstanceCreateInfo *, const VkAllocationCallbacks *, VkInstance *instance)
{
    if (!g.device) {
        g.device = MTL::CreateSystemDefaultDevice();
        if (!g.device || !g.device->supportsRaytracing()) {
            vkmtl_error("no Metal device with ray tracing support\n");
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        g.queue = g.device->newCommandQueue();

        MTL::ResidencySetDescriptor *desc = MTL::ResidencySetDescriptor::alloc()->init();
        NS::Error *error = nullptr;
        g.residency = g.device->newResidencySet(desc, &error);
        desc->release();
        if (!g.residency) {
            vkmtl_error("couldn't create a residency set\n");
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        g.queue->addResidencySet(g.residency);
        g.next_address = 1ull << 32;
    }
    g.offscreen = Cvar_VariableInteger("vid_hidden") != 0;
    *instance = reinterpret_cast<VkInstance>(&the_instance);
    return VK_SUCCESS;
}

void vkDestroyInstance(VkInstance, const VkAllocationCallbacks *)
{
}

VkResult vkEnumeratePhysicalDevices(VkInstance, uint32_t *count, VkPhysicalDevice *devices)
{
    if (devices && *count >= 1)
        devices[0] = reinterpret_cast<VkPhysicalDevice>(&the_physical_device);
    *count = 1;
    return VK_SUCCESS;
}

VkResult vkEnumeratePhysicalDeviceGroups(VkInstance, uint32_t *count, VkPhysicalDeviceGroupProperties *)
{
    *count = 0;
    return VK_SUCCESS;
}

void vkGetPhysicalDeviceProperties(VkPhysicalDevice, VkPhysicalDeviceProperties *props)
{
    memset(props, 0, sizeof(*props));
    props->apiVersion = VK_API_VERSION_1_2;
    props->driverVersion = 1;
    props->vendorID = 0x106b;
    props->deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    strncpy(props->deviceName, g.device->name()->utf8String(), sizeof(props->deviceName) - 1);
    props->limits.timestampPeriod = 1.0f;
    props->limits.minUniformBufferOffsetAlignment = 256;
    props->limits.minStorageBufferOffsetAlignment = 256;
    props->limits.minTexelBufferOffsetAlignment = 256;
    props->limits.minMemoryMapAlignment = 64;
    props->limits.nonCoherentAtomSize = 256;
    props->limits.optimalBufferCopyOffsetAlignment = 1;
    props->limits.optimalBufferCopyRowPitchAlignment = 1;
    props->limits.maxImageDimension2D = 16384;
    props->limits.maxImageDimension3D = 2048;
    props->limits.maxImageDimensionCube = 16384;
    props->limits.maxImageArrayLayers = 2048;
    props->limits.maxSamplerAnisotropy = 16.0f;
    props->limits.maxPushConstantsSize = 256;
    props->limits.maxBoundDescriptorSets = 4;
    props->limits.lineWidthRange[0] = props->limits.lineWidthRange[1] = 1.0f;
}

void vkGetPhysicalDeviceProperties2(VkPhysicalDevice device, VkPhysicalDeviceProperties2 *props)
{
    vkGetPhysicalDeviceProperties(device, &props->properties);

    for (VkBaseOutStructure *s = (VkBaseOutStructure *)props->pNext; s; s = s->pNext) {
        switch ((int)s->sType) {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR: {
            auto *p = (VkPhysicalDeviceAccelerationStructurePropertiesKHR *)s;
            p->maxGeometryCount = 1 << 24;
            p->maxInstanceCount = 1 << 24;
            p->maxPrimitiveCount = 1 << 28;
            p->minAccelerationStructureScratchOffsetAlignment = 256;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR: {
            auto *p = (VkPhysicalDeviceRayTracingPipelinePropertiesKHR *)s;
            p->shaderGroupHandleSize = 32;
            p->shaderGroupBaseAlignment = 64;
            p->shaderGroupHandleAlignment = 32;
            p->maxRayRecursionDepth = 1;
            break;
        }
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES: {
            auto *p = (VkPhysicalDeviceDriverProperties *)s;
            p->driverID = VK_DRIVER_ID_MOLTENVK;
            strncpy(p->driverName, "Metal", sizeof(p->driverName) - 1);
            strncpy(p->driverInfo, "vkpt on Metal", sizeof(p->driverInfo) - 1);
            break;
        }
        }
    }
}

static void set_bools(void *base, size_t header, size_t size, VkBool32 value)
{
    VkBool32 *b = (VkBool32 *)((char *)base + header);
    for (size_t i = 0; i < (size - header) / sizeof(VkBool32); i++)
        b[i] = value;
}

void vkGetPhysicalDeviceFeatures(VkPhysicalDevice, VkPhysicalDeviceFeatures *features)
{
    set_bools(features, 0, sizeof(*features), VK_TRUE);
}

void vkGetPhysicalDeviceFeatures2(VkPhysicalDevice device, VkPhysicalDeviceFeatures2 *features)
{
    vkGetPhysicalDeviceFeatures(device, &features->features);

    const size_t header = offsetof(VkPhysicalDeviceVulkan12Features, samplerMirrorClampToEdge);
    for (VkBaseOutStructure *s = (VkBaseOutStructure *)features->pNext; s; s = s->pNext) {
        switch ((int)s->sType) {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
            set_bools(s, header, sizeof(VkPhysicalDeviceVulkan12Features), VK_TRUE);
            // the fp16 shader variants are not needed
            ((VkPhysicalDeviceVulkan12Features *)s)->shaderFloat16 = VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES:
            set_bools(s, header, sizeof(VkPhysicalDevice16BitStorageFeatures), VK_FALSE);
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES:
            set_bools(s, header, sizeof(VkPhysicalDeviceLineRasterizationFeatures), VK_FALSE);
            break;
        }
    }
}

void vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties *props)
{
    memset(props, 0, sizeof(*props));
    props->memoryTypeCount = 1;
    props->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    props->memoryHeapCount = 1;
    props->memoryHeaps[0].size = g.device->recommendedMaxWorkingSetSize();
    props->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
}

void vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice, uint32_t *count, VkQueueFamilyProperties *props)
{
    if (props && *count >= 1) {
        memset(props, 0, sizeof(*props));
        props->queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
        props->queueCount = 1;
        props->timestampValidBits = 64;
    }
    *count = 1;
}

VkResult vkCreateDevice(VkPhysicalDevice, const VkDeviceCreateInfo *, const VkAllocationCallbacks *, VkDevice *device)
{
    *device = reinterpret_cast<VkDevice>(&the_device);
    return VK_SUCCESS;
}

void vkDestroyDevice(VkDevice, const VkAllocationCallbacks *)
{
    wait_idle();
}

void vkGetDeviceQueue(VkDevice, uint32_t, uint32_t, VkQueue *queue)
{
    *queue = reinterpret_cast<VkQueue>(&the_queue);
}

VkResult vkDeviceWaitIdle(VkDevice)
{
    wait_idle();
    return VK_SUCCESS;
}

VkResult vkQueueWaitIdle(VkQueue)
{
    wait_idle();
    return VK_SUCCESS;
}

// ------------------------------------------------------------------------------ memory

VkResult vkAllocateMemory(VkDevice, const VkMemoryAllocateInfo *info, const VkAllocationCallbacks *, VkDeviceMemory *out)
{
    Memory *memory = new Memory();
    memory->size = info->allocationSize;
    memory->address = g.next_address;
    g.next_address += (info->allocationSize + 0xffffull) & ~0xffffull;
    g.next_address += 0x10000;
    g.memories[memory->address] = memory;
    *out = reinterpret_cast<VkDeviceMemory>(memory);
    return VK_SUCCESS;
}

void vkFreeMemory(VkDevice, VkDeviceMemory handle, const VkAllocationCallbacks *)
{
    Memory *memory = VKMTL_HANDLE(Memory, handle);
    if (!memory)
        return;
    g.memories.erase(memory->address);
    if (memory->buffer) {
        residency_remove(memory->buffer);
        memory->buffer->release();
    }
    free(memory->image_mapping);
    delete memory;
}

VkResult vkMapMemory(VkDevice, VkDeviceMemory handle, VkDeviceSize offset, VkDeviceSize, VkMemoryMapFlags, void **data)
{
    Memory *memory = VKMTL_HANDLE(Memory, handle);

    // A linear image is read back by mapping its memory: copy the texture out.
    if (memory->image) {
        Image *image = memory->image;
        wait_idle();
        size_t row = (size_t)image->info.extent.width * format_bytes_per_pixel(image->info.format);
        free(memory->image_mapping);
        memory->image_mapping = malloc(row * image->info.extent.height);
        image->texture->getBytes(memory->image_mapping, row,
                                 MTL::Region::Make2D(0, 0, image->info.extent.width, image->info.extent.height), 0);
        *data = (char *)memory->image_mapping + offset;
        return VK_SUCCESS;
    }

    *data = (char *)memory->get()->contents() + offset;
    return VK_SUCCESS;
}

void vkUnmapMemory(VkDevice, VkDeviceMemory)
{
}

// ------------------------------------------------------------------------------ buffers

VkResult vkCreateBuffer(VkDevice, const VkBufferCreateInfo *info, const VkAllocationCallbacks *, VkBuffer *out)
{
    Buffer *buffer = new Buffer();
    buffer->size = info->size;
    buffer->usage = info->usage;
    *out = reinterpret_cast<VkBuffer>(buffer);
    return VK_SUCCESS;
}

void vkDestroyBuffer(VkDevice, VkBuffer handle, const VkAllocationCallbacks *)
{
    delete VKMTL_HANDLE(Buffer, handle);
}

void vkGetBufferMemoryRequirements(VkDevice, VkBuffer handle, VkMemoryRequirements *reqs)
{
    reqs->size = (VKMTL_HANDLE(Buffer, handle)->size + 255) & ~(VkDeviceSize)255;
    reqs->alignment = 256;
    reqs->memoryTypeBits = 1;
}

VkResult vkBindBufferMemory(VkDevice, VkBuffer handle, VkDeviceMemory memory, VkDeviceSize offset)
{
    Buffer *buffer = VKMTL_HANDLE(Buffer, handle);
    buffer->memory = VKMTL_HANDLE(Memory, memory);
    buffer->offset = offset;
    return VK_SUCCESS;
}

VkResult vkBindBufferMemory2(VkDevice device, uint32_t count, const VkBindBufferMemoryInfo *infos)
{
    for (uint32_t i = 0; i < count; i++)
        vkBindBufferMemory(device, infos[i].buffer, infos[i].memory, infos[i].memoryOffset);
    return VK_SUCCESS;
}

VkDeviceAddress vkGetBufferDeviceAddress(VkDevice, const VkBufferDeviceAddressInfo *info)
{
    Buffer *buffer = VKMTL_HANDLE(Buffer, info->buffer);
    return buffer->memory->address + buffer->offset;
}

VkResult vkCreateBufferView(VkDevice, const VkBufferViewCreateInfo *info, const VkAllocationCallbacks *, VkBufferView *out)
{
    Buffer *buffer = VKMTL_HANDLE(Buffer, info->buffer);
    uint32_t texel = format_bytes_per_pixel(info->format);
    VkDeviceSize range = info->range == VK_WHOLE_SIZE ? buffer->size - info->offset : info->range;
    NS::UInteger width = range / texel;

    MTL::TextureDescriptor *desc = MTL::TextureDescriptor::textureBufferDescriptor(
        pixel_format(info->format), width, MTL::ResourceStorageModeShared, MTL::TextureUsageShaderRead);

    BufferView *view = new BufferView();
    view->buffer = buffer;
    view->texture = buffer->mtl()->newTexture(desc, buffer->offset + info->offset, width * texel);
    if (!view->texture) {
        vkmtl_error("couldn't create a texture buffer of %lu texels\n", (unsigned long)width);
        delete view;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    *out = reinterpret_cast<VkBufferView>(view);
    return VK_SUCCESS;
}

void vkDestroyBufferView(VkDevice, VkBufferView handle, const VkAllocationCallbacks *)
{
    BufferView *view = VKMTL_HANDLE(BufferView, handle);
    if (!view)
        return;
    view->texture->release();
    delete view;
}

// ------------------------------------------------------------------------------ images

static MTL::Texture *create_texture(const VkImageCreateInfo *info, bool swapchain)
{
    MTL::TextureDescriptor *desc = MTL::TextureDescriptor::alloc()->init();

    bool cube = (info->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) && info->arrayLayers == 6;
    if (info->imageType == VK_IMAGE_TYPE_3D)
        desc->setTextureType(MTL::TextureType3D);
    else if (cube)
        desc->setTextureType(MTL::TextureTypeCube);
    else if (info->arrayLayers > 1)
        desc->setTextureType(MTL::TextureType2DArray);
    else
        desc->setTextureType(MTL::TextureType2D);

    desc->setPixelFormat(pixel_format(info->format));
    desc->setWidth(info->extent.width);
    desc->setHeight(info->extent.height);
    desc->setDepth(info->imageType == VK_IMAGE_TYPE_3D ? info->extent.depth : 1);
    desc->setMipmapLevelCount(info->mipLevels);
    if (!cube && info->imageType != VK_IMAGE_TYPE_3D)
        desc->setArrayLength(info->arrayLayers);

    // Everything can be read by a shader or copied; the rest follows the Vulkan usage.
    MTL::TextureUsage usage = MTL::TextureUsageShaderRead;
    if (info->usage & VK_IMAGE_USAGE_STORAGE_BIT)
        usage |= MTL::TextureUsageShaderWrite;
    if (info->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                       | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        usage |= MTL::TextureUsageRenderTarget;  // clears and scaled blits are render passes
    if (is_rgba8(info->format))
        usage |= MTL::TextureUsagePixelFormatView;  // sRGB and linear views of the same image
    desc->setUsage(usage);

    // Linear images are the ones vkpt maps to read pixels back.
    desc->setStorageMode(info->tiling == VK_IMAGE_TILING_LINEAR ? MTL::StorageModeShared : MTL::StorageModePrivate);

    MTL::Texture *texture = g.device->newTexture(desc);
    desc->release();
    if (texture)
        residency_add(texture);
    else
        vkmtl_error("couldn't create a %ux%u texture\n", info->extent.width, info->extent.height);
    (void)swapchain;
    return texture;
}

VkResult vkCreateImage(VkDevice, const VkImageCreateInfo *info, const VkAllocationCallbacks *, VkImage *out)
{
    Image *image = new Image();
    image->info = *info;
    image->info.pNext = nullptr;
    image->texture = create_texture(info, false);
    if (!image->texture) {
        delete image;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    g.images.push_back(image);
    *out = reinterpret_cast<VkImage>(image);
    return VK_SUCCESS;
}

void vkDestroyImage(VkDevice, VkImage handle, const VkAllocationCallbacks *)
{
    Image *image = VKMTL_HANDLE(Image, handle);
    if (!image || image->is_swapchain)
        return;
    for (size_t i = 0; i < g.images.size(); i++) {
        if (g.images[i] == image) {
            g.images.erase(g.images.begin() + i);
            break;
        }
    }
    residency_remove(image->texture);
    image->texture->release();
    delete image;
}

void vkGetImageMemoryRequirements(VkDevice, VkImage handle, VkMemoryRequirements *reqs)
{
    const VkImageCreateInfo &info = VKMTL_HANDLE(Image, handle)->info;
    VkDeviceSize size = (VkDeviceSize)info.extent.width * info.extent.height * info.extent.depth
        * info.arrayLayers * format_bytes_per_pixel(info.format);
    if (info.mipLevels > 1)
        size += size / 3;
    reqs->size = (size + 4095) & ~(VkDeviceSize)4095;
    reqs->alignment = 4096;
    reqs->memoryTypeBits = 1;
}

VkResult vkBindImageMemory(VkDevice, VkImage handle, VkDeviceMemory memory, VkDeviceSize)
{
    Image *image = VKMTL_HANDLE(Image, handle);
    if (image->info.tiling == VK_IMAGE_TILING_LINEAR)
        VKMTL_HANDLE(Memory, memory)->image = image;
    return VK_SUCCESS;
}

VkResult vkBindImageMemory2(VkDevice device, uint32_t count, const VkBindImageMemoryInfo *infos)
{
    for (uint32_t i = 0; i < count; i++)
        vkBindImageMemory(device, infos[i].image, infos[i].memory, infos[i].memoryOffset);
    return VK_SUCCESS;
}

void vkGetImageSubresourceLayout(VkDevice, VkImage handle, const VkImageSubresource *, VkSubresourceLayout *layout)
{
    const VkImageCreateInfo &info = VKMTL_HANDLE(Image, handle)->info;
    layout->offset = 0;
    layout->rowPitch = (VkDeviceSize)info.extent.width * format_bytes_per_pixel(info.format);
    layout->size = layout->rowPitch * info.extent.height;
    layout->arrayPitch = layout->size;
    layout->depthPitch = layout->size;
}

VkResult vkCreateImageView(VkDevice, const VkImageViewCreateInfo *info, const VkAllocationCallbacks *, VkImageView *out)
{
    Image *image = VKMTL_HANDLE(Image, info->image);
    const VkImageSubresourceRange &range = info->subresourceRange;

    uint32_t levels = range.levelCount == VK_REMAINING_MIP_LEVELS
        ? image->info.mipLevels - range.baseMipLevel : range.levelCount;
    uint32_t layers = range.layerCount == VK_REMAINING_ARRAY_LAYERS
        ? image->info.arrayLayers - range.baseArrayLayer : range.layerCount;

    MTL::TextureType type;
    switch (info->viewType) {
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: type = MTL::TextureType2DArray; break;
    case VK_IMAGE_VIEW_TYPE_CUBE:     type = MTL::TextureTypeCube; layers = 6; break;
    case VK_IMAGE_VIEW_TYPE_3D:       type = MTL::TextureType3D; layers = 1; break;
    default:                          type = MTL::TextureType2D; layers = 1; break;
    }

    ImageView *view = new ImageView();
    view->image = image;
    view->range = range;
    view->range.levelCount = levels;
    view->range.layerCount = layers;
    view->texture = image->texture->newTextureView(pixel_format(info->format), type,
        NS::Range::Make(range.baseMipLevel, levels), NS::Range::Make(range.baseArrayLayer, layers));
    if (!view->texture) {
        vkmtl_error("couldn't create an image view\n");
        delete view;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    *out = reinterpret_cast<VkImageView>(view);
    return VK_SUCCESS;
}

void vkDestroyImageView(VkDevice, VkImageView handle, const VkAllocationCallbacks *)
{
    ImageView *view = VKMTL_HANDLE(ImageView, handle);
    if (!view)
        return;
    view->texture->release();
    delete view;
}

// ------------------------------------------------------------------------------ samplers

static MTL::SamplerAddressMode address_mode(VkSamplerAddressMode mode)
{
    switch (mode) {
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:   return MTL::SamplerAddressModeClampToEdge;
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: return MTL::SamplerAddressModeClampToBorderColor;
    case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: return MTL::SamplerAddressModeMirrorRepeat;
    default:                                      return MTL::SamplerAddressModeRepeat;
    }
}

VkResult vkCreateSampler(VkDevice, const VkSamplerCreateInfo *info, const VkAllocationCallbacks *, VkSampler *out)
{
    MTL::SamplerDescriptor *desc = MTL::SamplerDescriptor::alloc()->init();
    desc->setMagFilter(info->magFilter == VK_FILTER_NEAREST ? MTL::SamplerMinMagFilterNearest : MTL::SamplerMinMagFilterLinear);
    desc->setMinFilter(info->minFilter == VK_FILTER_NEAREST ? MTL::SamplerMinMagFilterNearest : MTL::SamplerMinMagFilterLinear);
    desc->setMipFilter(info->mipmapMode == VK_SAMPLER_MIPMAP_MODE_NEAREST ? MTL::SamplerMipFilterNearest : MTL::SamplerMipFilterLinear);
    desc->setSAddressMode(address_mode(info->addressModeU));
    desc->setTAddressMode(address_mode(info->addressModeV));
    desc->setRAddressMode(address_mode(info->addressModeW));
    if (info->anisotropyEnable && info->maxAnisotropy > 1.0f)
        desc->setMaxAnisotropy((NS::UInteger)(info->maxAnisotropy > 16.0f ? 16.0f : info->maxAnisotropy));
    desc->setLodMinClamp(info->minLod);
    desc->setLodMaxClamp(info->maxLod);
    switch (info->borderColor) {
    case VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE:
    case VK_BORDER_COLOR_INT_OPAQUE_WHITE:
        desc->setBorderColor(MTL::SamplerBorderColorOpaqueWhite);
        break;
    case VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK:
    case VK_BORDER_COLOR_INT_TRANSPARENT_BLACK:
        desc->setBorderColor(MTL::SamplerBorderColorTransparentBlack);
        break;
    default:
        desc->setBorderColor(MTL::SamplerBorderColorOpaqueBlack);
        break;
    }
    desc->setSupportArgumentBuffers(true);

    Sampler *sampler = new Sampler();
    sampler->state = g.device->newSamplerState(desc);
    desc->release();
    *out = reinterpret_cast<VkSampler>(sampler);
    return VK_SUCCESS;
}

void vkDestroySampler(VkDevice, VkSampler handle, const VkAllocationCallbacks *)
{
    Sampler *sampler = VKMTL_HANDLE(Sampler, handle);
    if (!sampler)
        return;
    sampler->state->release();
    delete sampler;
}

// ------------------------------------------------------------------------------ surface, swapchain

SDL_bool vkmtl_get_instance_extensions(SDL_Window *, unsigned int *count, const char **)
{
    *count = 0;
    return SDL_TRUE;
}

SDL_bool vkmtl_create_surface(SDL_Window *window, VkInstance, VkSurfaceKHR *out)
{
    SDL_MetalView view = SDL_Metal_CreateView(window);
    if (!view)
        return SDL_FALSE;

    Surface *surface = new Surface();
    surface->view = view;
    surface->layer = static_cast<CA::MetalLayer *>(SDL_Metal_GetLayer(view));
    surface->layer->setDevice(g.device);
    surface->layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
    surface->layer->setFramebufferOnly(false);
    surface_window = window;
    *out = reinterpret_cast<VkSurfaceKHR>(surface);
    return SDL_TRUE;
}

void vkDestroySurfaceKHR(VkInstance, VkSurfaceKHR handle, const VkAllocationCallbacks *)
{
    Surface *surface = VKMTL_HANDLE(Surface, handle);
    if (!surface)
        return;
    SDL_Metal_DestroyView(surface->view);
    delete surface;
}

VkResult vkGetPhysicalDeviceSurfaceSupportKHR(VkPhysicalDevice, uint32_t, VkSurfaceKHR, VkBool32 *supported)
{
    *supported = VK_TRUE;
    return VK_SUCCESS;
}

VkResult vkGetPhysicalDeviceSurfaceCapabilitiesKHR(VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilitiesKHR *caps)
{
    int width = 0, height = 0;
    SDL_Metal_GetDrawableSize(surface_window, &width, &height);

    memset(caps, 0, sizeof(*caps));
    caps->minImageCount = 1;
    caps->maxImageCount = 1;
    caps->currentExtent.width = width;
    caps->currentExtent.height = height;
    caps->minImageExtent = caps->currentExtent;
    caps->maxImageExtent = caps->currentExtent;
    caps->maxImageArrayLayers = 1;
    caps->supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    caps->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    caps->supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    caps->supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return VK_SUCCESS;
}

VkResult vkGetPhysicalDeviceSurfaceFormatsKHR(VkPhysicalDevice, VkSurfaceKHR, uint32_t *count, VkSurfaceFormatKHR *formats)
{
    if (formats && *count >= 1) {
        formats[0].format = VK_FORMAT_B8G8R8A8_SRGB;
        formats[0].colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    }
    *count = 1;
    return VK_SUCCESS;
}

VkResult vkGetPhysicalDeviceSurfacePresentModesKHR(VkPhysicalDevice, VkSurfaceKHR, uint32_t *count, VkPresentModeKHR *modes)
{
    if (modes && *count >= 2) {
        modes[0] = VK_PRESENT_MODE_FIFO_KHR;
        modes[1] = VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
    *count = 2;
    return VK_SUCCESS;
}

VkResult vkCreateSwapchainKHR(VkDevice, const VkSwapchainCreateInfoKHR *info, const VkAllocationCallbacks *, VkSwapchainKHR *out)
{
    Swapchain *swapchain = new Swapchain();
    swapchain->surface = VKMTL_HANDLE(Surface, info->surface);
    swapchain->extent = info->imageExtent;

    VkImageCreateInfo image_info = {};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = info->imageFormat;
    image_info.extent = { info->imageExtent.width, info->imageExtent.height, 1 };
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.usage = info->imageUsage | VK_IMAGE_USAGE_SAMPLED_BIT;

    swapchain->image.info = image_info;
    swapchain->image.is_swapchain = true;
    swapchain->image.texture = create_texture(&image_info, true);
    if (!swapchain->image.texture) {
        delete swapchain;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    CA::MetalLayer *layer = swapchain->surface->layer;
    layer->setDrawableSize(CGSizeMake(info->imageExtent.width, info->imageExtent.height));
    layer->setDisplaySyncEnabled(info->presentMode == VK_PRESENT_MODE_FIFO_KHR);

    *out = reinterpret_cast<VkSwapchainKHR>(swapchain);
    return VK_SUCCESS;
}

void vkDestroySwapchainKHR(VkDevice, VkSwapchainKHR handle, const VkAllocationCallbacks *)
{
    Swapchain *swapchain = VKMTL_HANDLE(Swapchain, handle);
    if (!swapchain)
        return;
    wait_idle();
    residency_remove(swapchain->image.texture);
    swapchain->image.texture->release();
    delete swapchain;
}

VkResult vkGetSwapchainImagesKHR(VkDevice, VkSwapchainKHR handle, uint32_t *count, VkImage *images)
{
    if (images && *count >= 1)
        images[0] = reinterpret_cast<VkImage>(&VKMTL_HANDLE(Swapchain, handle)->image);
    *count = 1;
    return VK_SUCCESS;
}

VkResult vkAcquireNextImageKHR(VkDevice, VkSwapchainKHR handle, uint64_t, VkSemaphore, VkFence, uint32_t *index)
{
    Swapchain *swapchain = VKMTL_HANDLE(Swapchain, handle);
    *index = 0;

    int width = 0, height = 0;
    SDL_Metal_GetDrawableSize(surface_window, &width, &height);
    if (width > 0 && height > 0
        && ((uint32_t)width != swapchain->extent.width || (uint32_t)height != swapchain->extent.height))
        return VK_ERROR_OUT_OF_DATE_KHR;
    return VK_SUCCESS;
}

VkResult vkAcquireNextImage2KHR(VkDevice device, const VkAcquireNextImageInfoKHR *info, uint32_t *index)
{
    return vkAcquireNextImageKHR(device, info->swapchain, info->timeout, info->semaphore, info->fence, index);
}

VkResult vkQueuePresentKHR(VkQueue, const VkPresentInfoKHR *info)
{
    Swapchain *swapchain = VKMTL_HANDLE(Swapchain, info->pSwapchains[0]);

    if (!g.offscreen) {
        CA::MetalDrawable *drawable = swapchain->surface->layer->nextDrawable();
        MTL::Texture *source = swapchain->image.texture;
        if (drawable && drawable->texture()->width() == source->width()
            && drawable->texture()->height() == source->height()) {
            MTL::CommandBuffer *cmd = g.queue->commandBuffer();
            MTL::BlitCommandEncoder *blit = cmd->blitCommandEncoder();
            blit->copyFromTexture(source, drawable->texture());
            blit->endEncoding();
            cmd->presentDrawable(drawable);
            cmd->commit();
        }
    }

    debug_frame();
    profile_report();
    drain_pool();
    return VK_SUCCESS;
}

// ------------------------------------------------------------------------------ synchronization

VkResult vkCreateSemaphore(VkDevice, const VkSemaphoreCreateInfo *, const VkAllocationCallbacks *, VkSemaphore *out)
{
    static int dummy;
    *out = reinterpret_cast<VkSemaphore>(&dummy);
    return VK_SUCCESS;
}

void vkDestroySemaphore(VkDevice, VkSemaphore, const VkAllocationCallbacks *)
{
}

VkResult vkCreateFence(VkDevice, const VkFenceCreateInfo *info, const VkAllocationCallbacks *, VkFence *out)
{
    Fence *fence = new Fence();
    fence->signaled = (info->flags & VK_FENCE_CREATE_SIGNALED_BIT) != 0;
    *out = reinterpret_cast<VkFence>(fence);
    return VK_SUCCESS;
}

void vkDestroyFence(VkDevice, VkFence handle, const VkAllocationCallbacks *)
{
    Fence *fence = VKMTL_HANDLE(Fence, handle);
    if (!fence)
        return;
    if (fence->pending)
        fence->pending->release();
    delete fence;
}

VkResult vkWaitForFences(VkDevice, uint32_t count, const VkFence *fences, VkBool32, uint64_t)
{
    for (uint32_t i = 0; i < count; i++) {
        Fence *fence = VKMTL_HANDLE(Fence, fences[i]);
        if (fence->pending) {
            fence->pending->waitUntilCompleted();
            fence->pending->release();
            fence->pending = nullptr;
            fence->signaled = true;
        }
    }
    return VK_SUCCESS;
}

VkResult vkResetFences(VkDevice, uint32_t count, const VkFence *fences)
{
    for (uint32_t i = 0; i < count; i++) {
        Fence *fence = VKMTL_HANDLE(Fence, fences[i]);
        if (fence->pending) {
            fence->pending->release();
            fence->pending = nullptr;
        }
        fence->signaled = false;
    }
    return VK_SUCCESS;
}

// ------------------------------------------------------------------------------ queries, debug

// Timestamps are not collected: the profiler reads zeroes.
VkResult vkCreateQueryPool(VkDevice, const VkQueryPoolCreateInfo *, const VkAllocationCallbacks *, VkQueryPool *out)
{
    static int dummy;
    *out = reinterpret_cast<VkQueryPool>(&dummy);
    return VK_SUCCESS;
}

void vkDestroyQueryPool(VkDevice, VkQueryPool, const VkAllocationCallbacks *)
{
}

VkResult vkGetQueryPoolResults(VkDevice, VkQueryPool, uint32_t, uint32_t, size_t size, void *data, VkDeviceSize, VkQueryResultFlags)
{
    memset(data, 0, size);
    return VK_SUCCESS;
}

static VkResult create_debug_messenger(VkInstance, const VkDebugUtilsMessengerCreateInfoEXT *,
                                       const VkAllocationCallbacks *, VkDebugUtilsMessengerEXT *out)
{
    static int dummy;
    *out = reinterpret_cast<VkDebugUtilsMessengerEXT>(&dummy);
    return VK_SUCCESS;
}

static void destroy_debug_messenger(VkInstance, VkDebugUtilsMessengerEXT, const VkAllocationCallbacks *)
{
}

static void cmd_begin_label(VkCommandBuffer, const VkDebugUtilsLabelEXT *)
{
}

static void cmd_end_label(VkCommandBuffer)
{
}

// Acceleration structure entry points, defined in vk_metal_commands.cpp.
VkResult vkCreateAccelerationStructureKHR(VkDevice, const VkAccelerationStructureCreateInfoKHR *,
                                          const VkAllocationCallbacks *, VkAccelerationStructureKHR *);
void vkDestroyAccelerationStructureKHR(VkDevice, VkAccelerationStructureKHR, const VkAllocationCallbacks *);
void vkCmdBuildAccelerationStructuresKHR(VkCommandBuffer, uint32_t, const VkAccelerationStructureBuildGeometryInfoKHR *,
                                         const VkAccelerationStructureBuildRangeInfoKHR *const *);
VkDeviceAddress vkGetAccelerationStructureDeviceAddressKHR(VkDevice, const VkAccelerationStructureDeviceAddressInfoKHR *);
void vkGetAccelerationStructureBuildSizesKHR(VkDevice, VkAccelerationStructureBuildTypeKHR,
                                             const VkAccelerationStructureBuildGeometryInfoKHR *, const uint32_t *,
                                             VkAccelerationStructureBuildSizesInfoKHR *);

static VkResult set_object_name(VkDevice, const VkDebugMarkerObjectNameInfoEXT *info)
{
    if (info->objectType == VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT && info->object && info->pObjectName) {
        Image *image = reinterpret_cast<Image *>(info->object);
        image->name = info->pObjectName;
        image->texture->setLabel(nsstr(info->pObjectName));
    }
    return VK_SUCCESS;
}

// Loaded by vkpt but never called.
static void cmd_copy_accel(VkCommandBuffer, const VkCopyAccelerationStructureInfoKHR *)
{
    vkmtl_unsupported("copying acceleration structures");
}

static void cmd_write_accel_properties(VkCommandBuffer, uint32_t, const VkAccelerationStructureKHR *, VkQueryType,
                                       VkQueryPool, uint32_t)
{
    vkmtl_unsupported("querying acceleration structure properties");
}

static PFN_vkVoidFunction find_function(const char *name)
{
    static const struct {
        const char *name;
        PFN_vkVoidFunction function;
    } functions[] = {
        { "vkCreateDebugUtilsMessengerEXT", (PFN_vkVoidFunction)create_debug_messenger },
        { "vkDestroyDebugUtilsMessengerEXT", (PFN_vkVoidFunction)destroy_debug_messenger },
        { "vkCmdBeginDebugUtilsLabelEXT", (PFN_vkVoidFunction)cmd_begin_label },
        { "vkCmdEndDebugUtilsLabelEXT", (PFN_vkVoidFunction)cmd_end_label },
        { "vkCreateAccelerationStructureKHR", (PFN_vkVoidFunction)vkCreateAccelerationStructureKHR },
        { "vkDestroyAccelerationStructureKHR", (PFN_vkVoidFunction)vkDestroyAccelerationStructureKHR },
        { "vkCmdBuildAccelerationStructuresKHR", (PFN_vkVoidFunction)vkCmdBuildAccelerationStructuresKHR },
        { "vkGetAccelerationStructureDeviceAddressKHR", (PFN_vkVoidFunction)vkGetAccelerationStructureDeviceAddressKHR },
        { "vkGetAccelerationStructureBuildSizesKHR", (PFN_vkVoidFunction)vkGetAccelerationStructureBuildSizesKHR },
        { "vkGetBufferDeviceAddress", (PFN_vkVoidFunction)vkGetBufferDeviceAddress },
        { "vkDebugMarkerSetObjectNameEXT", (PFN_vkVoidFunction)set_object_name },
        { "vkCmdCopyAccelerationStructureKHR", (PFN_vkVoidFunction)cmd_copy_accel },
        { "vkCmdWriteAccelerationStructuresPropertiesKHR", (PFN_vkVoidFunction)cmd_write_accel_properties },
    };
    for (const auto &f : functions) {
        if (!strcmp(f.name, name))
            return f.function;
    }
    return nullptr;
}

PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance, const char *name)
{
    return find_function(name);
}

PFN_vkVoidFunction vkGetDeviceProcAddr(VkDevice, const char *name)
{
    return find_function(name);
}

}  // extern "C"

// Used by vk_metal_commands.cpp.
namespace vkmtl {

static float half_to_float(uint16_t h)
{
    int exponent = (h >> 10) & 0x1f, mantissa = h & 0x3ff;
    float value;
    if (exponent == 0)
        value = mantissa * (1.0f / (1 << 24));
    else if (exponent == 31)
        value = mantissa ? __builtin_nanf("") : __builtin_inff();
    else
        value = __builtin_ldexpf((float)(mantissa | 0x400), exponent - 25);
    return (h & 0x8000) ? -value : value;
}

void debug_frame(void)
{
    static int frame, target = -2;
    if (target == -2) {
        const char *env = getenv("VKMTL_DUMP_FRAME");
        target = env ? atoi(env) : -1;
    }
    extern std::atomic<long long> gpu_microseconds;
    static int stats = -1;
    if (stats < 0)
        stats = getenv("VKMTL_STATS") != nullptr;
    if (stats && frame % 30 == 29) {
        static double last;
        double now = CACurrentMediaTime();
        vkmtl_print("frame %d: %.1f ms GPU per frame, %.1f frames per second\n", frame,
                    gpu_microseconds.exchange(0) / 30000.0, last > 0.0 ? 30.0 / (now - last) : 0.0);
        last = now;
    }
    if (frame++ != target)
        return;

    wait_idle();
    for (Image *image : g.images) {
        const VkImageCreateInfo &info = image->info;
        if (image->name.empty() || info.imageType != VK_IMAGE_TYPE_2D || info.arrayLayers != 1)
            continue;

        uint32_t bpp = format_bytes_per_pixel(info.format);
        size_t row = (size_t)info.extent.width * bpp, size = row * info.extent.height;
        MTL::Buffer *staging = g.device->newBuffer(size, MTL::ResourceStorageModeShared);
        MTL::CommandBuffer *cmd = g.queue->commandBuffer();
        MTL::BlitCommandEncoder *blit = cmd->blitCommandEncoder();
        blit->copyFromTexture(image->texture, 0, 0, MTL::Origin::Make(0, 0, 0),
                              MTL::Size::Make(info.extent.width, info.extent.height, 1), staging, 0, row, size);
        blit->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();

        const uint8_t *bytes = static_cast<const uint8_t *>(staging->contents());
        size_t pixels = (size_t)info.extent.width * info.extent.height, nonzero = 0, bad = 0;
        double sum[4] = {};
        bool is_half = info.format == VK_FORMAT_R16G16B16A16_SFLOAT || info.format == VK_FORMAT_R16G16_SFLOAT
            || info.format == VK_FORMAT_R16_SFLOAT;
        int channels = is_half ? bpp / 2 : 0;
        for (size_t p = 0; p < pixels; p++) {
            const uint8_t *px = bytes + p * bpp;
            bool any = false;
            for (uint32_t b = 0; b < bpp; b++)
                any |= px[b] != 0;
            nonzero += any;
            for (int c = 0; c < channels; c++) {
                float v = half_to_float(((const uint16_t *)px)[c]);
                if (v != v || v > 1e30f || v < -1e30f)
                    bad++;
                else
                    sum[c] += v;
            }
        }
        vkmtl_print("image %-32s %4ux%-4u nonzero %5.1f%%", image->name.c_str(), info.extent.width,
                    info.extent.height, 100.0 * nonzero / pixels);
        if (channels)
            vkmtl_print("  mean %.4g %.4g %.4g %.4g  nan/inf %zu", sum[0] / pixels, sum[1] / pixels,
                        sum[2] / pixels, sum[3] / pixels, bad);
        vkmtl_print("\n");
        staging->release();
    }
}

void submitted(MTL::CommandBuffer *cmd)
{
    if (g.last_submitted)
        g.last_submitted->release();
    g.last_submitted = cmd->retain();
}

}  // namespace vkmtl
