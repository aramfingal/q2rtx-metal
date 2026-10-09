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

// metal.h -- internal declarations shared by the Metal renderer's C files.

#pragma once

#include "shared/shared.h"
#include "common/common.h"
#include "common/cvar.h"
#include "common/cmd.h"
#include "common/files.h"
#include "client/client.h"
#include "client/video.h"
#include "refresh/refresh.h"
#include "refresh/images.h"
#include "refresh/models.h"

#include "mtl_api.h"

// Alias model mesh, laid out as vkpt's so the vkpt model loaders and path tracer
// code can be shared with this renderer as they are ported.
typedef struct maliasframe_s {
    vec3_t  scale;
    vec3_t  translate;
    vec3_t  bounds[2];
    vec_t   radius;
} maliasframe_t;

typedef struct maliasmesh_s {
    int             numverts;
    int             numtris;
    int             numindices;
    int             numskins;
    int             tri_offset;
    int             *indices;
    vec3_t          *positions;
    vec3_t          *normals;
    vec2_t          *tex_coords;
    vec3_t          *tangents;
    uint32_t        *blend_indices;
    uint32_t        *blend_weights;
    struct pbr_material_s **materials;
    bool            handedness;
} maliasmesh_t;

// draw.c
void MTL_Draw_RegisterFunctions(void);
void MTL_Draw_EndFrame(void);
void MTL_Draw_DiscardRawPic(void);

// textures.c
extern cvar_t *cvar_pt_bilerp_chars;
extern cvar_t *cvar_pt_bilerp_pics;
void MTL_Textures_Init(void);
void MTL_Textures_UpdateFilters(void);
void IMG_Load_Metal(image_t *image, byte *pic);
void IMG_Unload_Metal(image_t *image);

// world.c
void MTL_World_Load(const char *name);
void MTL_World_Free(void);
void MTL_World_RenderView(const refdef_t *fd);

// models.c
int MOD_LoadMD2_Metal(model_t *model, const void *rawdata, size_t length, const char *mod_name);
int MOD_LoadMD3_Metal(model_t *model, const void *rawdata, size_t length, const char *mod_name);
int MOD_LoadIQM_Metal(model_t *model, const void *rawdata, size_t length, const char *mod_name);
void MOD_Reference_Metal(model_t *model);
