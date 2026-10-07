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

// textures.c -- image upload hooks. Each image_t is uploaded into the Metal texture
// slot with its r_images index, so a qhandle_t can be used as a texture slot directly.

#include "metal.h"

cvar_t *cvar_pt_bilerp_chars;
cvar_t *cvar_pt_bilerp_pics;
static cvar_t *cvar_pt_nearest;

// Texture filtering follows vkpt: UI chars and pics use the pt_bilerp_* cvars,
// everything else pt_nearest (0 = linear magnification).
static mtl_filter_t filter_for_image(const image_t *image)
{
    switch (image->type) {
    case IT_FONT:
        return cvar_pt_bilerp_chars->integer ? MTL_FILTER_LINEAR : MTL_FILTER_NEAREST;
    case IT_PIC:
        return cvar_pt_bilerp_pics->integer ? MTL_FILTER_LINEAR : MTL_FILTER_NEAREST;
    default:
        return cvar_pt_nearest->integer ? MTL_FILTER_NEAREST : MTL_FILTER_LINEAR;
    }
}

void MTL_Textures_UpdateFilters(void)
{
    for (int i = 0; i < r_numImages; i++) {
        image_t *image = &r_images[i];
        if (image->registration_sequence)
            mtl_texture_set_filter(i, filter_for_image(image));
    }
}

static void filter_cvar_changed(cvar_t *self)
{
    MTL_Textures_UpdateFilters();
}

void MTL_Textures_Init(void)
{
    cvar_pt_nearest = Cvar_Get("pt_nearest", "0", CVAR_ARCHIVE);
    cvar_pt_bilerp_chars = Cvar_Get("pt_bilerp_chars", "0", CVAR_ARCHIVE);
    cvar_pt_bilerp_pics = Cvar_Get("pt_bilerp_pics", "0", CVAR_ARCHIVE);
    cvar_pt_nearest->changed = filter_cvar_changed;
    cvar_pt_bilerp_chars->changed = filter_cvar_changed;
    cvar_pt_bilerp_pics->changed = filter_cvar_changed;
}

void IMG_Load_Metal(image_t *image, byte *pic)
{
    // Keep the pixels, as vkpt does: material and emissive light extraction read them.
    image->pix_data = pic;

    uint32_t slot = image - r_images;
    mtl_format_t format = image->pixel_format == PF_R16_UNORM ? MTL_FORMAT_R16 : MTL_FORMAT_RGBA8;
    bool mipmaps = image->type != IT_PIC && image->type != IT_FONT;

    if (!mtl_texture_upload(slot, image->upload_width, image->upload_height, format, pic,
                            image->is_srgb, mipmaps, filter_for_image(image))) {
        Com_DPrintf("%s: couldn't upload %s (%dx%d)\n", __func__, image->name,
                    image->upload_width, image->upload_height);
    }
}

void IMG_Unload_Metal(image_t *image)
{
    if (image->pix_data)
        Z_Free(image->pix_data);
    image->pix_data = NULL;

    mtl_texture_free(image - r_images);
}
