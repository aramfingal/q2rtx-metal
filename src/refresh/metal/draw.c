/*
Copyright (C) 2018 Christoph Schied
Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
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

// draw.c -- 2D drawing. A port of vkpt/draw.c: quads are queued in submission order
// and drawn by mtl_end_frame.

#include "metal.h"

#define MAX_STRETCH_PICS (1 << 14)

typedef struct {
    color_t colors[2]; // 0 - actual color, 1 - transparency (for text drawing)
    float scale;
    float alpha_scale;
} drawStatic_t;

static drawStatic_t draw = {
    .scale = 1.0f,
    .alpha_scale = 1.0f
};

static mtl_quad_t stretch_pic_queue[MAX_STRETCH_PICS];
static int num_stretch_pics;

static clipRect_t clip_rect;
static bool clip_enable;

static image_t *raw_image;

static inline void enqueue_stretch_pic(
        float x, float y, float w, float h,
        float s1, float t1, float s2, float t2,
        uint32_t color, uint32_t tex)
{
    if (draw.alpha_scale == 0.f)
        return;

    if (num_stretch_pics == MAX_STRETCH_PICS) {
        Com_EPrintf("Error: stretch pic queue full!\n");
        return;
    }

    if (clip_enable) {
        if (x >= clip_rect.right || x + w <= clip_rect.left || y >= clip_rect.bottom || y + h <= clip_rect.top)
            return;

        if (x < clip_rect.left) {
            float dw = clip_rect.left - x;
            s1 += dw / w * (s2 - s1);
            w -= dw;
            x = clip_rect.left;

            if (w <= 0) return;
        }

        if (x + w > clip_rect.right) {
            float dw = (x + w) - clip_rect.right;
            s2 -= dw / w * (s2 - s1);
            w -= dw;

            if (w <= 0) return;
        }

        if (y < clip_rect.top) {
            float dh = clip_rect.top - y;
            t1 += dh / h * (t2 - t1);
            h -= dh;
            y = clip_rect.top;

            if (h <= 0) return;
        }

        if (y + h > clip_rect.bottom) {
            float dh = (y + h) - clip_rect.bottom;
            t2 -= dh / h * (t2 - t1);
            h -= dh;

            if (h <= 0) return;
        }
    }

    mtl_quad_t *sp = stretch_pic_queue + num_stretch_pics++;

    float width = r_config.width * draw.scale;
    float height = r_config.height * draw.scale;

    sp->x = 2.0f * x / width - 1.0f;
    sp->y = 2.0f * y / height - 1.0f;
    sp->w = 2.0f * w / width;
    sp->h = 2.0f * h / height;

    sp->s = s1;
    sp->t = t1;
    sp->w_s = s2 - s1;
    sp->h_t = t2 - t1;

    if (draw.alpha_scale < 1.f) {
        float alpha = (color >> 24) & 0xff;
        alpha *= draw.alpha_scale;
        alpha = max(0.f, min(255.f, alpha));
        color = (color & 0xffffff) | ((int)(alpha) << 24);
    }

    sp->color = color;
    sp->tex = tex;
}

static uint32_t texture_for_handle(qhandle_t pic)
{
    // Handles are r_images indices; textures.c uploads each image into the same slot.
    // Freed images draw as white, as in vkpt.
    if (pic >= 0 && pic < MAX_RIMAGES && !r_images[pic].registration_sequence)
        return MTL_TEX_WHITE;
    return (uint32_t)pic;
}

void MTL_Draw_EndFrame(void)
{
    mtl_end_frame(stretch_pic_queue, num_stretch_pics);
    num_stretch_pics = 0;
}

static void R_SetClipRect_Metal(const clipRect_t *clip)
{
    if (clip) {
        clip_enable = true;
        clip_rect = *clip;
    } else {
        clip_enable = false;
    }
}

static void R_ClearColor_Metal(void)
{
    draw.colors[0].u32 = U32_WHITE;
    draw.colors[1].u32 = U32_WHITE;
}

static void R_SetAlpha_Metal(float alpha)
{
    alpha = powf(fabsf(alpha), 0.4545f); // un-sRGB the alpha
    draw.colors[0].u8[3] = draw.colors[1].u8[3] = alpha * 255;
}

static void R_SetAlphaScale_Metal(float alpha)
{
    draw.alpha_scale = alpha;
}

static void R_SetColor_Metal(uint32_t color)
{
    draw.colors[0].u32 = color;
    draw.colors[1].u8[3] = draw.colors[0].u8[3];
}

static void R_SetScale_Metal(float scale)
{
    draw.scale = scale;
}

static void R_DrawStretchPic_Metal(int x, int y, int w, int h, qhandle_t pic)
{
    enqueue_stretch_pic(
        x,    y,    w,    h,
        0.0f, 0.0f, 1.0f, 1.0f,
        draw.colors[0].u32, texture_for_handle(pic));
}

static void R_DrawPic_Metal(int x, int y, qhandle_t pic)
{
    image_t *image = IMG_ForHandle(pic);
    R_DrawStretchPic_Metal(x, y, image->width, image->height, pic);
}

static void R_DrawStretchRaw_Metal(int x, int y, int w, int h)
{
    if (!raw_image)
        return;
    R_DrawStretchPic_Metal(x, y, w, h, raw_image - r_images);
}

static void R_UpdateRawPic_Metal(int pic_w, int pic_h, const uint32_t *pic)
{
    if (raw_image)
        R_UnregisterImage(raw_image - r_images);

    size_t raw_size = pic_w * pic_h * 4;
    byte *raw_data = Z_Malloc(raw_size);
    memcpy(raw_data, pic, raw_size);
    static int raw_id;
    raw_image = r_images + R_RegisterRawImage(va("**raw[%d]**", raw_id++), pic_w, pic_h, raw_data, IT_SPRITE, IF_SRGB);
}

void MTL_Draw_DiscardRawPic(void)
{
    if (raw_image) {
        R_UnregisterImage(raw_image - r_images);
        raw_image = NULL;
    }
}

static void R_DrawKeepAspectPic_Metal(int x, int y, int w, int h, qhandle_t pic)
{
    image_t *image = IMG_ForHandle(pic);

    if (image->flags & IF_SCRAP) {
        R_DrawStretchPic_Metal(x, y, w, h, pic);
        return;
    }

    float scale_w = w;
    float scale_h = h * image->aspect;
    float scale = max(scale_w, scale_h);

    float s = (1.0f - scale_w / scale) * 0.5f;
    float t = (1.0f - scale_h / scale) * 0.5f;

    enqueue_stretch_pic(x, y, w, h, s, t, 1.0f - s, 1.0f - t, draw.colors[0].u32, texture_for_handle(pic));
}

#define DIV64 (1.0f / 64.0f)

static void R_TileClear_Metal(int x, int y, int w, int h, qhandle_t pic)
{
    enqueue_stretch_pic(x, y, w, h,
        x * DIV64, y * DIV64, (x + w) * DIV64, (y + h) * DIV64,
        U32_WHITE, texture_for_handle(pic));
}

static void R_DrawFill8_Metal(int x, int y, int w, int h, int c)
{
    if (!w || !h)
        return;
    enqueue_stretch_pic(x, y, w, h, 0.0f, 0.0f, 1.0f, 1.0f,
        d_8to24table[c & 0xff], MTL_TEX_WHITE);
}

static void R_DrawFill32_Metal(int x, int y, int w, int h, uint32_t color)
{
    if (!w || !h)
        return;
    enqueue_stretch_pic(x, y, w, h, 0.0f, 0.0f, 1.0f, 1.0f,
        color, MTL_TEX_WHITE);
}

static inline void draw_char(int x, int y, int flags, int c, qhandle_t font)
{
    if ((c & 127) == 32) {
        return;
    }

    if (flags & UI_ALTCOLOR) {
        c |= 0x80;
    }
    if (flags & UI_XORCOLOR) {
        c ^= 0x80;
    }

    float s = (c & 15) * 0.0625f;
    float t = (c >> 4) * 0.0625f;

    float eps = 1e-5f; /* fixes some ugly artifacts */

    enqueue_stretch_pic(x, y, CHAR_WIDTH, CHAR_HEIGHT,
        s + eps, t + eps, s + 0.0625f - eps, t + 0.0625f - eps,
        draw.colors[c >> 7].u32, texture_for_handle(font));
}

static void R_DrawChar_Metal(int x, int y, int flags, int c, qhandle_t font)
{
    draw_char(x, y, flags, c & 255, font);
}

static int R_DrawString_Metal(int x, int y, int flags, size_t maxlen, const char *s, qhandle_t font)
{
    while (maxlen-- && *s) {
        byte c = *s++;
        draw_char(x, y, flags, c, font);
        x += CHAR_WIDTH;
    }

    return x;
}

void MTL_Draw_RegisterFunctions(void)
{
    R_ClearColor = R_ClearColor_Metal;
    R_SetAlpha = R_SetAlpha_Metal;
    R_SetAlphaScale = R_SetAlphaScale_Metal;
    R_SetColor = R_SetColor_Metal;
    R_SetClipRect = R_SetClipRect_Metal;
    R_SetScale = R_SetScale_Metal;
    R_DrawChar = R_DrawChar_Metal;
    R_DrawString = R_DrawString_Metal;
    R_DrawPic = R_DrawPic_Metal;
    R_DrawStretchPic = R_DrawStretchPic_Metal;
    R_DrawKeepAspectPic = R_DrawKeepAspectPic_Metal;
    R_DrawStretchRaw = R_DrawStretchRaw_Metal;
    R_TileClear = R_TileClear_Metal;
    R_DrawFill8 = R_DrawFill8_Metal;
    R_DrawFill32 = R_DrawFill32_Metal;
    R_UpdateRawPic = R_UpdateRawPic_Metal;
    R_DiscardRawPic = MTL_Draw_DiscardRawPic;
}
