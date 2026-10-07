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

// main.c -- Metal renderer entry points (the macOS counterpart of vkpt/main.c).
//
// Current state: device, swapchain, 2D drawing, textures and screenshots work. The 3D
// view is not drawn yet; R_RenderFrame leaves it black until the path tracer is ported.

#include "metal.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_metal.h>

extern SDL_Window *get_sdl_window(void);

static SDL_MetalView metal_view;

static cvar_t *cvar_vsync;

// Read by client/tent.c whenever the renderer reports REF_TYPE_VKPT.
cvar_t *cvar_pt_beam_lights;

static void vsync_changed(cvar_t *self)
{
    mtl_set_vsync(self->integer != 0);
}

static ref_type_t R_Init_Metal(bool total)
{
    registration_sequence = 1;

    if (!vid.init(GAPI_METAL)) {
        Com_Error(ERR_FATAL, "VID_Init failed\n");
        return REF_TYPE_NONE;
    }

    metal_view = SDL_Metal_CreateView(get_sdl_window());
    if (!metal_view) {
        Com_Error(ERR_FATAL, "SDL_Metal_CreateView failed: %s\n", SDL_GetError());
        return REF_TYPE_NONE;
    }

    cvar_vsync = Cvar_Get("vid_vsync", "0", CVAR_ARCHIVE);
    cvar_vsync->changed = vsync_changed;
    cvar_pt_beam_lights = Cvar_Get("pt_beam_lights", "1.0", 0);

    MTL_Textures_Init();

    IMG_Init();
    IMG_GetPalette();
    MOD_Init();

    if (!mtl_init(SDL_Metal_GetLayer(metal_view), r_config.width, r_config.height)) {
        Com_Error(ERR_FATAL, "Couldn't initialize Metal.\n");
        return REF_TYPE_NONE;
    }

    mtl_set_vsync(cvar_vsync->integer != 0);

    // The client treats this renderer as the RTX one (path-traced effects, MD3 models).
    return REF_TYPE_VKPT;
}

static void R_Shutdown_Metal(bool total)
{
    mtl_wait_idle();

    MTL_Draw_DiscardRawPic();
    IMG_FreeAll();
    mtl_shutdown();

    IMG_Shutdown();
    MOD_Shutdown();

    if (metal_view) {
        SDL_Metal_DestroyView(metal_view);
        metal_view = NULL;
    }

    vid.shutdown();
}

static void R_BeginFrame_Metal(void)
{
}

static void R_EndFrame_Metal(void)
{
    MTL_Draw_EndFrame();
}

static void R_ModeChanged_Metal(int width, int height, int flags)
{
    Com_DPrintf("mode changed %d %d\n", width, height);

    r_config.width  = width;
    r_config.height = height;
    r_config.flags  = flags;

    mtl_set_drawable_size(width, height);
}

static void R_RenderFrame_Metal(refdef_t *fd)
{
    // The path tracer is not ported yet; the 3D view stays black.
}

static void R_LightPoint_Metal(const vec3_t origin, vec3_t light)
{
    VectorSet(light, 1, 1, 1);
}

static void R_BeginRegistration_Metal(const char *name)
{
    registration_sequence++;
    Com_Printf("loading %s\n", name);

    Com_AddConfigFile("maps/default.cfg", 0);
    Com_AddConfigFile(va("maps/%s.cfg", name), 0);
}

static void R_EndRegistration_Metal(void)
{
    IMG_FreeUnused();
    MOD_FreeUnused();
}

static void R_SetSky_Metal(const char *name, float rotate, int autorotate, const vec3_t axis)
{
}

static void R_AddDecal_Metal(decal_t *d)
{
}

static bool R_InterceptKey_Metal(unsigned key, bool down)
{
    return false;
}

static bool R_IsHDR_Metal(void)
{
    return false;
}

static bool R_SupportsDebugLines_Metal(void)
{
    return false;
}

static void R_AddDebugText_Metal(const vec3_t origin, const vec3_t angles, const char *text,
                                 float size, uint32_t color, uint32_t time, bool depth_test)
{
}

static void IMG_ReadPixels_Metal(screenshot_t *s)
{
    int width = r_config.width;
    int height = r_config.height;
    int pitch = width * 3;

    s->pixels = FS_AllocTempMem(pitch * height);
    s->width = width;
    s->height = height;
    s->rowbytes = pitch;
    s->bpp = 3;

    mtl_wait_idle();
    if (!mtl_read_last_frame_rgb(s->pixels, width, height)) {
        Com_WPrintf("IMG_ReadPixels: no frame to read back yet\n");
        memset(s->pixels, 0, pitch * height);
    }
}

static void IMG_ReadPixelsHDR_Metal(screenshot_t *s)
{
    // HDR output is not supported yet (R_IsHDR returns false, so this is never called).
}

void R_RegisterFunctionsMetal(void)
{
    R_Init = R_Init_Metal;
    R_Shutdown = R_Shutdown_Metal;
    R_BeginRegistration = R_BeginRegistration_Metal;
    R_EndRegistration = R_EndRegistration_Metal;
    R_SetSky = R_SetSky_Metal;
    R_RenderFrame = R_RenderFrame_Metal;
    R_LightPoint = R_LightPoint_Metal;
    R_SupportsDebugLines = R_SupportsDebugLines_Metal;
    R_AddDebugText_ = R_AddDebugText_Metal;
    R_BeginFrame = R_BeginFrame_Metal;
    R_EndFrame = R_EndFrame_Metal;
    R_ModeChanged = R_ModeChanged_Metal;
    R_AddDecal = R_AddDecal_Metal;
    R_InterceptKey = R_InterceptKey_Metal;
    R_IsHDR = R_IsHDR_Metal;
    MTL_Draw_RegisterFunctions();
    IMG_Load = IMG_Load_Metal;
    IMG_Unload = IMG_Unload_Metal;
    IMG_ReadPixels = IMG_ReadPixels_Metal;
    IMG_ReadPixelsHDR = IMG_ReadPixelsHDR_Metal;
    MOD_LoadMD2 = MOD_LoadMD2_Metal;
    MOD_LoadMD3 = MOD_LoadMD3_Metal;
    MOD_LoadIQM = MOD_LoadIQM_Metal;
    MOD_Reference = MOD_Reference_Metal;
}
