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

// world.c -- BSP world geometry and the per-frame scene.
//
// The BSP's faces are triangulated into one unindexed triangle list, grouped by BSP
// model: model 0 is the static world, the others are inline models (doors, platforms)
// that entities place in the scene each frame. This is the simple counterpart of
// vkpt/bsp_mesh.c; materials, lights and the sky are not extracted yet.

#include "metal.h"
#include "common/bsp.h"

static bsp_t *world_bsp;

static bool face_is_drawn(const mface_t *face)
{
    int flags = face->texinfo->c.flags;

    if (flags & SURF_NODRAW)
        return false;
    // Glass is skipped until transparency is ported; water stays as an opaque surface.
    if ((flags & SURF_TRANS_MASK) && !(flags & SURF_WARP))
        return false;
    return face->numsurfedges >= 3;
}

static const float *face_vertex(const mface_t *face, int i)
{
    const msurfedge_t *surfedge = &face->firstsurfedge[i];
    return surfedge->edge->v[surfedge->vert]->point;
}

void MTL_World_Free(void)
{
    mtl_world_free();

    if (world_bsp) {
        BSP_Free(world_bsp);
        world_bsp = NULL;
    }
}

void MTL_World_Load(const char *name)
{
    char path[MAX_QPATH];
    bsp_t *bsp;

    MTL_World_Free();

    Q_concat(path, sizeof(path), "maps/", name, ".bsp");
    int ret = BSP_Load(path, &bsp);
    if (!bsp)
        Com_Error(ERR_DROP, "%s: couldn't load %s: %s", __func__, path, Q_ErrorString(ret));
    world_bsp = bsp;

    image_t **images = Z_Malloc(bsp->numtexinfo * sizeof(image_t *));
    for (int i = 0; i < bsp->numtexinfo; i++) {
        mtexinfo_t *texinfo = &bsp->texinfo[i];
        if (texinfo->c.flags & (SURF_NODRAW | SURF_SKY)) {
            images[i] = NULL;
            continue;
        }
        Q_concat(path, sizeof(path), "textures/", texinfo->name, ".wal");
        images[i] = IMG_Find(path, IT_WALL, IF_SRGB);
    }

    int num_triangles = 0;
    for (int i = 0; i < bsp->numfaces; i++) {
        if (face_is_drawn(&bsp->faces[i]))
            num_triangles += bsp->faces[i].numsurfedges - 2;
    }

    vec3_t *positions = Z_Malloc(num_triangles * 3 * sizeof(vec3_t));
    mtl_triangle_t *triangles = Z_Malloc(num_triangles * sizeof(mtl_triangle_t));
    mtl_model_t *models = Z_Malloc(bsp->nummodels * sizeof(mtl_model_t));

    // The world is every face that no inline model claims (as in vkpt's collect_surfaces).
    bool *inline_face = Z_Mallocz(bsp->numfaces * sizeof(bool));
    for (int m = 1; m < bsp->nummodels; m++) {
        const mmodel_t *model = &bsp->models[m];
        for (int f = 0; f < model->numfaces; f++)
            inline_face[model->firstface + f - bsp->faces] = true;
    }

    int tri = 0;
    for (int m = 0; m < bsp->nummodels; m++) {
        const mface_t *faces = m ? bsp->models[m].firstface : bsp->faces;
        int num_faces = m ? bsp->models[m].numfaces : bsp->numfaces;
        models[m].first_triangle = tri;

        for (int f = 0; f < num_faces; f++) {
            const mface_t *face = &faces[f];
            if (!face_is_drawn(face) || (!m && inline_face[f]))
                continue;

            const mtexinfo_t *texinfo = face->texinfo;
            const image_t *image = images[texinfo - bsp->texinfo];
            float inv_width = image && image->width ? 1.0f / image->width : 1.0f;
            float inv_height = image && image->height ? 1.0f / image->height : 1.0f;

            // triangle fan around the face's first vertex
            for (int i = 2; i < face->numsurfedges; i++, tri++) {
                const int corners[3] = { 0, i - 1, i };
                mtl_triangle_t *out = &triangles[tri];

                for (int c = 0; c < 3; c++) {
                    const float *p = face_vertex(face, corners[c]);
                    VectorCopy(p, positions[tri * 3 + c]);
                    out->uv[c][0] = (DotProduct(p, texinfo->axis[0]) + texinfo->offset[0]) * inv_width;
                    out->uv[c][1] = (DotProduct(p, texinfo->axis[1]) + texinfo->offset[1]) * inv_height;
                }

                out->tex = image ? (uint32_t)(image - r_images) : MTL_TEX_WHITE;
                out->flags = (texinfo->c.flags & SURF_SKY) ? MTL_TRI_SKY : 0;
            }
        }

        models[m].num_triangles = tri - models[m].first_triangle;
    }

    if (!mtl_world_upload((const float *)positions, triangles, tri, models, bsp->nummodels))
        Com_WPrintf("Couldn't build acceleration structures for %s\n", name);
    else
        Com_DPrintf("%s: %d triangles in %d models\n", name, tri, bsp->nummodels);

    Z_Free(inline_face);
    Z_Free(models);
    Z_Free(triangles);
    Z_Free(positions);
    Z_Free(images);
}

void MTL_World_RenderView(const refdef_t *fd)
{
    static mtl_instance_t instances[MTL_MAX_INSTANCES];
    int num_instances = 0;
    mtl_view_t view;

    if (!world_bsp || (fd->rdflags & RDF_NOWORLDMODEL))
        return;

    // the static world
    mtl_instance_t *world = &instances[num_instances++];
    world->model = 0;
    VectorClear(world->origin);
    VectorSet(world->axis[0], 1, 0, 0);
    VectorSet(world->axis[1], 0, 1, 0);
    VectorSet(world->axis[2], 0, 0, 1);

    // inline models
    for (int i = 0; i < fd->num_entities && num_instances < MTL_MAX_INSTANCES; i++) {
        const entity_t *ent = &fd->entities[i];
        if (!(ent->model & 0x80000000))
            continue;

        int index = ~ent->model;
        if (index < 1 || index >= world_bsp->nummodels)
            continue;

        mtl_instance_t *inst = &instances[num_instances++];
        inst->model = index;
        VectorCopy(ent->origin, inst->origin);
        AnglesToAxis(ent->angles, inst->axis);
    }

    VectorCopy(fd->vieworg, view.origin);
    AngleVectors(fd->viewangles, view.forward, view.right, view.up);
    view.tan_half_fov_x = tanf(DEG2RAD(fd->fov_x) * 0.5f);
    view.tan_half_fov_y = tanf(DEG2RAD(fd->fov_y) * 0.5f);
    view.instances = instances;
    view.num_instances = num_instances;

    mtl_render_view(&view);
}
