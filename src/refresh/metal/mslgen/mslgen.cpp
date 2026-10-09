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

// mslgen -- build tool that translates one SPIR-V shader to the Metal Shading Language.
//
//   mslgen in.spv out.metal
//
// Every descriptor set becomes a Metal argument buffer at the buffer index of its set
// number. The tool assigns the [[id]] of each resource itself and records them in a
// comment block at the top of the output, which the Vulkan-on-Metal layer
// (vk_metal_pipeline.cpp) reads to know where a Vulkan (set, binding) lives in the
// shader's argument buffer:
//
//   /*VKMTL
//   name <shader file name>
//   stage <comp|vert|frag>
//   entry <function name>
//   local <x> <y> <z>                                    (compute: threads per group)
//   push <buffer index>                                  (only with push constants)
//   bind <set> <binding> <count> <id> <sampler id> <runtime>
//   */
//
// <id> is the argument index of the buffer, texture or acceleration structure (the first
// one for an array), <sampler id> that of the sampler of a combined image sampler, or -1.
// A runtime-sized array (<runtime> = 1) is always the last member of its set.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "spirv_msl.hpp"

using namespace spirv_cross;

// Buffer indices used by the generated shaders; vk_metal.cpp uses the same numbers.
enum {
    PUSH_CONSTANT_BUFFER_INDEX = 8,
};

struct binding_t {
    uint32_t set, binding, count;
    int id, sampler_id;
    bool runtime;
    SPIRType::BaseType basetype;
    bool combined;
};

static bool read_file(const char *path, std::vector<uint32_t> &words)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return false;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    words.resize(size / sizeof(uint32_t));
    bool ok = fread(words.data(), sizeof(uint32_t), words.size(), f) == words.size();
    fclose(f);
    return ok;
}

static bool write_file(const char *path, const std::string &text)
{
    FILE *f = fopen(path, "wb");
    if (!f)
        return false;
    bool ok = fwrite(text.data(), 1, text.size(), f) == text.size();
    fclose(f);
    return ok;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s in.spv out.metal\n", argv[0]);
        return 2;
    }

    std::vector<uint32_t> words;
    if (!read_file(argv[1], words)) {
        fprintf(stderr, "mslgen: can't read %s\n", argv[1]);
        return 1;
    }

    try {
        CompilerMSL msl(std::move(words));

        CompilerMSL::Options options = msl.get_msl_options();
        options.platform = CompilerMSL::Options::macOS;
        options.set_msl_version(3, 1);
        options.argument_buffers = true;
        options.argument_buffers_tier = CompilerMSL::Options::ArgumentBuffersTier::Tier2;
        options.force_native_arrays = true;
        options.texture_buffer_native = true;
        msl.set_msl_options(options);

        // Vulkan's clip space has y pointing down.
        CompilerGLSL::Options common = msl.get_common_options();
        common.vertex.flip_vert_y = true;
        msl.set_common_options(common);

        spv::ExecutionModel model = msl.get_execution_model();
        const char *stage = model == spv::ExecutionModelVertex ? "vert"
                          : model == spv::ExecutionModelFragment ? "frag" : "comp";

        ShaderResources res = msl.get_shader_resources();

        std::vector<binding_t> bindings;
        auto collect = [&](const SmallVector<Resource> &list, bool combined) {
            for (const Resource &r : list) {
                const SPIRType &type = msl.get_type(r.type_id);
                binding_t b = {};
                b.set = msl.get_decoration(r.id, spv::DecorationDescriptorSet);
                b.binding = msl.get_decoration(r.id, spv::DecorationBinding);
                b.count = 1;
                if (!type.array.empty()) {
                    b.count = type.array.back();
                    b.runtime = b.count == 0;
                }
                b.basetype = type.basetype;
                b.combined = combined;
                b.id = b.sampler_id = -1;
                bindings.push_back(b);
            }
        };
        collect(res.uniform_buffers, false);
        collect(res.storage_buffers, false);
        collect(res.sampled_images, true);
        collect(res.storage_images, false);
        collect(res.separate_images, false);
        collect(res.separate_samplers, false);
        collect(res.acceleration_structures, false);

        std::sort(bindings.begin(), bindings.end(), [](const binding_t &a, const binding_t &b) {
            if (a.set != b.set)
                return a.set < b.set;
            if (a.runtime != b.runtime)
                return b.runtime;  // runtime arrays go last
            return a.binding < b.binding;
        });

        std::map<uint32_t, int> next_id;
        for (binding_t &b : bindings) {
            int &next = next_id[b.set];
            if (b.runtime && &b != &bindings.back() && (&b)[1].set == b.set) {
                fprintf(stderr, "mslgen: %s: set %u has more than one runtime array\n", argv[1], b.set);
                return 1;
            }
            b.id = next;
            next += b.runtime ? 1 : b.count;
            if (b.combined) {
                if (b.runtime) {
                    fprintf(stderr, "mslgen: %s: runtime array of combined image samplers "
                            "(set %u, binding %u)\n", argv[1], b.set, b.binding);
                    return 1;
                }
                b.sampler_id = next;
                next += b.count;
            }

            MSLResourceBinding rb;
            rb.stage = model;
            rb.basetype = b.basetype;
            rb.desc_set = b.set;
            rb.binding = b.binding;
            rb.count = b.count;
            rb.msl_buffer = rb.msl_texture = b.id;
            rb.msl_sampler = b.combined ? b.sampler_id : b.id;
            msl.add_msl_resource_binding(rb);
        }

        for (const auto &set : next_id) {
            MSLResourceBinding rb;
            rb.stage = model;
            rb.desc_set = set.first;
            rb.binding = kArgumentBufferBinding;
            rb.msl_buffer = set.first;
            msl.add_msl_resource_binding(rb);
            msl.set_argument_buffer_device_address_space(set.first, true);
        }

        if (!res.push_constant_buffers.empty()) {
            MSLResourceBinding rb;
            rb.stage = model;
            rb.desc_set = kPushConstDescSet;
            rb.binding = kPushConstBinding;
            rb.msl_buffer = PUSH_CONSTANT_BUFFER_INDEX;
            msl.add_msl_resource_binding(rb);
        }

        std::string source = msl.compile();

        std::string bind;
        std::string name = argv[1];
        name = name.substr(name.find_last_of('/') + 1);
        name = name.substr(0, name.rfind(".spv"));
        bind += "name " + name + "\n";
        bind += std::string("stage ") + stage + "\n";
        bind += "entry " + msl.get_cleansed_entry_point_name("main", model) + "\n";
        if (model == spv::ExecutionModelGLCompute) {
            bind += "local";
            for (uint32_t i = 0; i < 3; i++)
                bind += " " + std::to_string(msl.get_execution_mode_argument(spv::ExecutionModeLocalSize, i));
            bind += "\n";
        }
        if (!res.push_constant_buffers.empty())
            bind += "push " + std::to_string(PUSH_CONSTANT_BUFFER_INDEX) + "\n";
        for (const binding_t &b : bindings) {
            char line[128];
            snprintf(line, sizeof(line), "bind %u %u %u %d %d %d\n", b.set, b.binding, b.count,
                     b.id, b.sampler_id, b.runtime ? 1 : 0);
            bind += line;
        }

        if (!write_file(argv[2], "/*VKMTL\n" + bind + "*/\n" + source)) {
            fprintf(stderr, "mslgen: can't write output for %s\n", argv[1]);
            return 1;
        }
    } catch (const std::exception &e) {
        fprintf(stderr, "mslgen: %s: %s\n", argv[1], e.what());
        return 1;
    }

    return 0;
}
