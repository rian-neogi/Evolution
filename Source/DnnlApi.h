#pragma once
// Minimal oneDNN v3 C ABI for runtime-only installations. All objects are opaque;
// this does not depend on private library layouts or generated platform headers.
// Adapted from oneDNN v3.9.1 include/oneapi/dnnl/{dnnl,dnnl_types,dnnl_common,
// dnnl_common_types}.h: https://github.com/oneapi-src/oneDNN/tree/v3.9.1/include/oneapi/dnnl
// Copyright 2016-2025 Intel Corporation. SPDX-License-Identifier: Apache-2.0
// License: ../ThirdParty/oneDNN-LICENSE
#ifdef EVOLUTION_DNNL_HEADERS
#include <oneapi/dnnl/dnnl.h>
#else
#include <cstddef>
#include <cstdint>
extern "C" {
enum dnnl_status_t { dnnl_success = 0, dnnl_out_of_memory = 1, dnnl_invalid_arguments = 2,
    dnnl_unimplemented = 3, dnnl_last_impl_reached = 4, dnnl_runtime_error = 5,
    dnnl_not_required = 6, dnnl_invalid_graph = 7, dnnl_invalid_graph_op = 8,
    dnnl_invalid_shape = 9, dnnl_invalid_data_type = 10 };
enum dnnl_data_type_t { dnnl_data_type_undef = 0, dnnl_f16 = 1, dnnl_bf16 = 2,
    dnnl_f32 = 3, dnnl_s32 = 4, dnnl_s8 = 5, dnnl_u8 = 6, dnnl_f64 = 7,
    dnnl_boolean = 8, dnnl_f8_e5m2 = 9, dnnl_f8_e4m3 = 10, dnnl_s4 = 11,
    dnnl_u4 = 12, dnnl_e8m0 = 13, dnnl_f4_e2m1 = 14, dnnl_f4_e3m0 = 15 };
enum dnnl_engine_kind_t { dnnl_any_engine = 0, dnnl_cpu = 1, dnnl_gpu = 2 };
using dnnl_dim_t = std::int64_t;
using dnnl_engine_t = struct dnnl_engine*;
using dnnl_stream_t = struct dnnl_stream*;
using dnnl_memory_desc_t = struct dnnl_memory_desc*;
using dnnl_memory_t = struct dnnl_memory*;
using dnnl_primitive_desc_t = struct dnnl_primitive_desc*;
using dnnl_primitive_t = struct dnnl_primitive*;
struct dnnl_primitive_attr;
struct dnnl_exec_arg_t { int arg; dnnl_memory_t memory; };
struct dnnl_version_t { int major, minor, patch; const char* hash; unsigned cpu_runtime, gpu_runtime; };
const dnnl_version_t* dnnl_version();
dnnl_status_t dnnl_engine_create(dnnl_engine_t*, dnnl_engine_kind_t, std::size_t);
dnnl_status_t dnnl_engine_destroy(dnnl_engine_t);
dnnl_status_t dnnl_stream_create(dnnl_stream_t*, dnnl_engine_t, unsigned);
dnnl_status_t dnnl_stream_wait(dnnl_stream_t);
dnnl_status_t dnnl_stream_destroy(dnnl_stream_t);
dnnl_status_t dnnl_memory_desc_create_with_strides(dnnl_memory_desc_t*, int,
    const dnnl_dim_t*, dnnl_data_type_t, const dnnl_dim_t*);
dnnl_status_t dnnl_memory_desc_destroy(dnnl_memory_desc_t);
dnnl_status_t dnnl_memory_create(dnnl_memory_t*, const dnnl_memory_desc*, dnnl_engine_t, void*);
dnnl_status_t dnnl_memory_destroy(dnnl_memory_t);
dnnl_status_t dnnl_memory_set_data_handle(dnnl_memory_t, void*);
dnnl_status_t dnnl_matmul_primitive_desc_create(dnnl_primitive_desc_t*, dnnl_engine_t,
    const dnnl_memory_desc*, const dnnl_memory_desc*, const dnnl_memory_desc*,
    const dnnl_memory_desc*, const dnnl_primitive_attr*);
dnnl_status_t dnnl_primitive_desc_destroy(dnnl_primitive_desc_t);
dnnl_status_t dnnl_primitive_create(dnnl_primitive_t*, const dnnl_primitive_desc*);
dnnl_status_t dnnl_primitive_destroy(dnnl_primitive_t);
dnnl_status_t dnnl_primitive_execute(const dnnl_primitive*, dnnl_stream_t, int, const dnnl_exec_arg_t*);
}
#endif
