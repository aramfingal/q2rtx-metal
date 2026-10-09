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

// vk_metal.h -- the part of the Metal renderer that vkpt sees besides <vulkan/vulkan.h>.
//
// On macOS the path tracer in refresh/vkpt runs on Metal. The vk_metal_*.cpp files
// implement the subset of the Vulkan API that vkpt uses directly on Metal (metal-cpp),
// and the shaders are the vkpt shaders translated to the Metal Shading Language at
// build time (see mslgen). These functions replace the SDL Vulkan window hooks.

#pragma once

#include <vulkan/vulkan.h>
#include <SDL2/SDL.h>

#ifdef __cplusplus
extern "C" {
#endif

// Stand-ins for SDL_Vulkan_GetInstanceExtensions and SDL_Vulkan_CreateSurface.
SDL_bool vkmtl_get_instance_extensions(SDL_Window *window, unsigned int *count, const char **names);
SDL_bool vkmtl_create_surface(SDL_Window *window, VkInstance instance, VkSurfaceKHR *surface);

#ifdef __cplusplus
}
#endif
