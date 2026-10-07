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

// models.c -- model loader hooks.
//
// Placeholder until the vkpt model loaders (vkpt/models.c) are ported: alias and IQM
// models register as empty models so the client gets valid handles and nothing is
// drawn for them. Sprites are loaded by the common code in refresh/models.c.

#include "metal.h"

static int load_empty(model_t *model)
{
    model->type = MOD_EMPTY;
    model->nummeshes = 0;
    model->numframes = 0;
    return Q_ERR_SUCCESS;
}

int MOD_LoadMD2_Metal(model_t *model, const void *rawdata, size_t length, const char *mod_name)
{
    return load_empty(model);
}

int MOD_LoadMD3_Metal(model_t *model, const void *rawdata, size_t length, const char *mod_name)
{
    return load_empty(model);
}

int MOD_LoadIQM_Metal(model_t *model, const void *rawdata, size_t length, const char *mod_name)
{
    return load_empty(model);
}

void MOD_Reference_Metal(model_t *model)
{
    model->registration_sequence = registration_sequence;

    if (model->type == MOD_SPRITE) {
        for (int i = 0; i < model->numframes; i++)
            model->spriteframes[i].image->registration_sequence = registration_sequence;
    }
}
