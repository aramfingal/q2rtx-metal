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

// mtl_api.h -- C interface to the Metal device layer (mtl_device.cpp).
//
// The renderer proper (main.c, draw.c, textures.c, models.c) is C and uses the
// engine headers directly. Everything that touches Metal lives behind this
// header in C++ (metal-cpp), so engine headers never have to compile as C++.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Frames the CPU may record ahead of the GPU. Per-frame buffers are ring-indexed by this.
#define MTL_MAX_FRAMES_IN_FLIGHT 3

// Texture slots are image indices (image - r_images), so this matches MAX_RIMAGES.
#define MTL_MAX_TEXTURES 8192

// Slot of the 1x1 white texture used for fills.
#define MTL_TEX_WHITE 0xffffffffu

// One 2D quad. Layout matches the vertex shader's Quad struct and vkpt's StretchPic_t.
// Position is in normalized device coordinates with y pointing down (as vkpt).
typedef struct {
    float x, y, w, h;
    float s, t, w_s, h_t;
    uint32_t color;  // RGBA8, sRGB-encoded (byte 0 = red)
    uint32_t tex;    // texture slot or MTL_TEX_WHITE
} mtl_quad_t;

typedef enum {
    MTL_FORMAT_RGBA8,
    MTL_FORMAT_R16,
} mtl_format_t;

typedef enum {
    MTL_FILTER_LINEAR,
    MTL_FILTER_NEAREST,
} mtl_filter_t;

// Device lifecycle. layer is the CAMetalLayer from SDL_Metal_GetLayer.
bool mtl_init(void *layer, int width, int height);
void mtl_shutdown(void);
const char *mtl_device_name(void);
bool mtl_supports_raytracing(void);

void mtl_set_drawable_size(int width, int height);
void mtl_set_vsync(bool enabled);

// Textures. Pixel data is copied; the caller keeps ownership.
bool mtl_texture_upload(uint32_t slot, int width, int height, mtl_format_t format,
                        const void *pixels, bool srgb, bool mipmaps, mtl_filter_t filter);
void mtl_texture_set_filter(uint32_t slot, mtl_filter_t filter);
void mtl_texture_free(uint32_t slot);
void mtl_texture_free_all(void);

// Frame. mtl_end_frame waits for a free in-flight slot, acquires a drawable, clears it,
// draws the queued 2D quads in order and presents. Returns false if no drawable was
// available (window minimized or occluded); the frame is then dropped.
bool mtl_end_frame(const mtl_quad_t *quads, int num_quads);

// Blocks until the GPU has finished all submitted work.
void mtl_wait_idle(void);

// Copies the last presented frame into dst as tightly packed RGB8, bottom row first
// (the layout the screenshot writers expect). Returns false if no frame is available.
bool mtl_read_last_frame_rgb(uint8_t *dst, int width, int height);

#ifdef __cplusplus
}
#endif
