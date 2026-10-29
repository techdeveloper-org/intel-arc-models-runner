/**
 * @file arc_level_zero_bridge.c
 * @brief Dynamic loader implementation for Level Zero USM and bounded-wait DMA queues.
 */

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "arc_level_zero_bridge.h"
#include "arc_ze_types.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
  #include <windows.h>
  #define LOAD_LIB(name) (void*)LoadLibraryA(name)
  #define FREE_LIB(h) FreeLibrary((HMODULE)(h))
#else
  #include <dlfcn.h>
  #include <time.h>
  #define LOAD_LIB(name) dlopen(name, RTLD_NOW)
  #define FREE_LIB(h) dlclose(h)
#endif

#define ARC_SYNC_SLICE_MS 250u
#define ARC_USM_ALIGNMENT 64u

typedef struct {
    pfn_zeInit zeInit;
    pfn_zeDriverGet zeDriverGet;
    pfn_zeDeviceGet zeDeviceGet;
    pfn_zeDeviceGetProperties zeDeviceGetProperties;
    pfn_zeContextCreate zeContextCreate;
    pfn_zeContextDestroy zeContextDestroy;
    pfn_zeCommandQueueCreate zeCommandQueueCreate;
    pfn_zeCommandQueueDestroy zeCommandQueueDestroy;
    pfn_zeCommandListCreate zeCommandListCreate;
    pfn_zeCommandListDestroy zeCommandListDestroy;
    pfn_zeCommandListClose zeCommandListClose;
    pfn_zeCommandListReset zeCommandListReset;
    pfn_zeCommandQueueExecuteCommandLists zeCommandQueueExecuteCommandLists;
    pfn_zeCommandQueueSynchronize zeCommandQueueSynchronize;
    pfn_zeMemAllocHost zeMemAllocHost;
    pfn_zeMemAllocDevice zeMemAllocDevice;
    pfn_zeMemAllocShared zeMemAllocShared;
    pfn_zeMemFree zeMemFree;
    pfn_zeCommandListAppendMemoryCopy zeCommandListAppendMemoryCopy;
} ArcZeApi;

typedef struct {
    void* ptr;
    size_t size;
    ArcUsmType type;
} ArcAllocRecord;

struct ArcZeContext {
    void* ze_module;
    void* driver_handle;
    void* device_handle;
    void* context_handle;
    void* queue_handle;
    void* cmdlist_handle;
    ArcZeApi api;
    uint32_t device_type;
    uint64_t sync_timeout_ms;
    atomic_int_fast32_t last_status;
    atomic_uint_fast32_t last_result;
    atomic_bool is_dead;
    atomic_uint_fast32_t pending_cmds;
    atomic_flag table_lock;
    uint32_t alloc_count;
    ArcAllocRecord allocs[ARC_MAX_TRACKED_ALLOCS];
};

/**
 * @brief Record a status and return it so callers can write "return arc_ze_set_status(...)".
 *
 * @param ctx Context to annotate; ignored when NULL.
 * @param status ArcZeStatus value.
 * @return The same status.
 */
static int arc_ze_set_status(ArcZeContext* ctx, int status) {
    if (ctx) {
        atomic_store_explicit(&ctx->last_status, status, memory_order_relaxed);
    }
    return status;
}

/**
 * @brief Record a driver result and map it to ARC_ZE_ERR_API.
 *
 * @param ctx Context to annotate.
 * @param result Raw ze_result_t from the driver.
 * @return ARC_ZE_ERR_API.
 */
static int arc_ze_api_error(ArcZeContext* ctx, ze_result_t result) {
    atomic_store_explicit(&ctx->last_result, result, memory_order_relaxed);
    return arc_ze_set_status(ctx, ARC_ZE_ERR_API);
}

/**
 * @brief Monotonic clock in milliseconds.
 *
 * @return Milliseconds from an arbitrary fixed origin.
 */
static uint64_t arc_now_ms(void) {
#if defined(_WIN32)
    return (uint64_t)GetTickCount64();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
#endif
}

/**
 * @brief Resolve a symbol into a function pointer without an object-to-function cast.
 *
 * ISO C does not define converting void* to a function pointer; copying the pointer bytes is the
 * portable form accepted by POSIX dlsym and Win32 GetProcAddress users.
 *
 * @param module Loaded module handle.
 * @param name Symbol name.
 * @param out Address of the function pointer to fill.
 * @param out_size sizeof(*out).
 * @return true if the symbol was found.
 */
static bool arc_bind(void* module, const char* name, void* out, size_t out_size) {
#if defined(_WIN32)
    FARPROC symbol = GetProcAddress((HMODULE)module, name);
#else
    void* symbol = dlsym(module, name);
#endif
    if (!symbol || out_size != sizeof symbol) {
        return false;
    }
    memcpy(out, &symbol, sizeof symbol);
    return true;
}

/**
 * @brief Resolve every required Level Zero entry point.
 *
 * @param ctx Context whose api table is filled.
 * @return true if all symbols were found.
 */
static bool arc_bind_all(ArcZeContext* ctx) {
    void* m = ctx->ze_module;
    ArcZeApi* a = &ctx->api;
    return arc_bind(m, "zeInit", &a->zeInit, sizeof a->zeInit)
        && arc_bind(m, "zeDriverGet", &a->zeDriverGet, sizeof a->zeDriverGet)
        && arc_bind(m, "zeDeviceGet", &a->zeDeviceGet, sizeof a->zeDeviceGet)
        && arc_bind(m, "zeDeviceGetProperties", &a->zeDeviceGetProperties, sizeof a->zeDeviceGetProperties)
        && arc_bind(m, "zeContextCreate", &a->zeContextCreate, sizeof a->zeContextCreate)
        && arc_bind(m, "zeContextDestroy", &a->zeContextDestroy, sizeof a->zeContextDestroy)
        && arc_bind(m, "zeCommandQueueCreate", &a->zeCommandQueueCreate, sizeof a->zeCommandQueueCreate)
        && arc_bind(m, "zeCommandQueueDestroy", &a->zeCommandQueueDestroy, sizeof a->zeCommandQueueDestroy)
        && arc_bind(m, "zeCommandListCreate", &a->zeCommandListCreate, sizeof a->zeCommandListCreate)
        && arc_bind(m, "zeCommandListDestroy", &a->zeCommandListDestroy, sizeof a->zeCommandListDestroy)
        && arc_bind(m, "zeCommandListClose", &a->zeCommandListClose, sizeof a->zeCommandListClose)
        && arc_bind(m, "zeCommandListReset", &a->zeCommandListReset, sizeof a->zeCommandListReset)
        && arc_bind(m, "zeCommandQueueExecuteCommandLists", &a->zeCommandQueueExecuteCommandLists, sizeof a->zeCommandQueueExecuteCommandLists)
        && arc_bind(m, "zeCommandQueueSynchronize", &a->zeCommandQueueSynchronize, sizeof a->zeCommandQueueSynchronize)
        && arc_bind(m, "zeMemAllocHost", &a->zeMemAllocHost, sizeof a->zeMemAllocHost)
        && arc_bind(m, "zeMemAllocDevice", &a->zeMemAllocDevice, sizeof a->zeMemAllocDevice)
        && arc_bind(m, "zeMemAllocShared", &a->zeMemAllocShared, sizeof a->zeMemAllocShared)
        && arc_bind(m, "zeMemFree", &a->zeMemFree, sizeof a->zeMemFree)
        && arc_bind(m, "zeCommandListAppendMemoryCopy", &a->zeCommandListAppendMemoryCopy, sizeof a->zeCommandListAppendMemoryCopy);
}

/**
 * @brief Load the Level Zero loader library, honouring ARC_ZE_LOADER_PATH first.
 *
 * @return Module handle, or NULL if no candidate could be loaded.
 */
static void* arc_load_loader(void) {
    const char* override_path = getenv(ARC_ZE_LOADER_ENV);
    if (override_path && override_path[0] != '\0') {
        return LOAD_LIB(override_path);
    }
#if defined(_WIN32)
    const char* candidates[] = {"ze_loader.dll", NULL};
#else
    const char* candidates[] = {"libze_loader.so.1", "libze_loader.so", NULL};
#endif
    for (int i = 0; candidates[i] != NULL; ++i) {
        void* module = LOAD_LIB(candidates[i]);
        if (module) {
            return module;
        }
    }
    return NULL;
}

/**
 * @brief Read the synchronize timeout from the environment, clamped below the TDR limit.
 *
 * @return Timeout in milliseconds in [1, ARC_ZE_MAX_SYNC_TIMEOUT_MS].
 */
static uint64_t arc_configured_sync_timeout_ms(void) {
    const char* text = getenv(ARC_ZE_SYNC_TIMEOUT_ENV);
    if (!text || text[0] == '\0') {
        return ARC_ZE_DEFAULT_SYNC_TIMEOUT_MS;
    }
    char* end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (end == text || *end != '\0' || value == 0) {
        return ARC_ZE_DEFAULT_SYNC_TIMEOUT_MS;
    }
    return value > ARC_ZE_MAX_SYNC_TIMEOUT_MS ? ARC_ZE_MAX_SYNC_TIMEOUT_MS : (uint64_t)value;
}

/**
 * @brief Scan all drivers and devices and select the first GPU.
 *
 * @param ctx Context; driver_handle, device_handle and device_type are set on success.
 * @return ARC_ZE_OK, ARC_ZE_ERR_NO_GPU, or ARC_ZE_ERR_API.
 */
static int arc_select_gpu(ArcZeContext* ctx) {
    uint32_t driver_count = 0;
    ze_result_t res = ctx->api.zeDriverGet(&driver_count, NULL);
    if (res != ZE_RESULT_SUCCESS) {
        return arc_ze_api_error(ctx, res);
    }
    if (driver_count == 0) {
        return ARC_ZE_ERR_NO_GPU;
    }

    void** drivers = (void**)calloc(driver_count, sizeof(void*));
    if (!drivers) {
        return ARC_ZE_ERR_API;
    }
    res = ctx->api.zeDriverGet(&driver_count, drivers);
    int status = res == ZE_RESULT_SUCCESS ? ARC_ZE_ERR_NO_GPU : arc_ze_api_error(ctx, res);

    for (uint32_t d = 0; status == ARC_ZE_ERR_NO_GPU && d < driver_count; ++d) {
        uint32_t device_count = 0;
        if (ctx->api.zeDeviceGet(drivers[d], &device_count, NULL) != ZE_RESULT_SUCCESS || device_count == 0) {
            continue;
        }
        void** devices = (void**)calloc(device_count, sizeof(void*));
        if (!devices) {
            continue;
        }
        if (ctx->api.zeDeviceGet(drivers[d], &device_count, devices) == ZE_RESULT_SUCCESS) {
            for (uint32_t i = 0; i < device_count; ++i) {
                ze_device_properties_t props;
                memset(&props, 0, sizeof props);
                props.stype = ZE_STYPE_DEVICE_PROPERTIES;
                if (ctx->api.zeDeviceGetProperties(devices[i], &props) == ZE_RESULT_SUCCESS && props.type == ZE_DEVICE_TYPE_GPU) {
                    ctx->driver_handle = drivers[d];
                    ctx->device_handle = devices[i];
                    ctx->device_type = props.type;
                    status = ARC_ZE_OK;
                    break;
                }
            }
        }
        free(devices);
    }
    free(drivers);
    return status;
}

/**
 * @brief Create the context, command queue and command list on the selected GPU.
 *
 * @param ctx Context with driver and device handles set.
 * @return ARC_ZE_OK or ARC_ZE_ERR_API with the failing driver result recorded.
 */
static int arc_create_objects(ArcZeContext* ctx) {
    ze_context_desc_t context_desc;
    memset(&context_desc, 0, sizeof context_desc);
    context_desc.stype = ZE_STYPE_CONTEXT_DESC;
    ze_result_t res = ctx->api.zeContextCreate(ctx->driver_handle, &context_desc, &ctx->context_handle);
    if (res != ZE_RESULT_SUCCESS) {
        return arc_ze_api_error(ctx, res);
    }

    ze_command_queue_desc_t queue_desc;
    memset(&queue_desc, 0, sizeof queue_desc);
    queue_desc.stype = ZE_STYPE_COMMAND_QUEUE_DESC;
    queue_desc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    res = ctx->api.zeCommandQueueCreate(ctx->context_handle, ctx->device_handle, &queue_desc, &ctx->queue_handle);
    if (res != ZE_RESULT_SUCCESS) {
        return arc_ze_api_error(ctx, res);
    }

    ze_command_list_desc_t list_desc;
    memset(&list_desc, 0, sizeof list_desc);
    list_desc.stype = ZE_STYPE_COMMAND_LIST_DESC;
    res = ctx->api.zeCommandListCreate(ctx->context_handle, ctx->device_handle, &list_desc, &ctx->cmdlist_handle);
    if (res != ZE_RESULT_SUCCESS) {
        return arc_ze_api_error(ctx, res);
    }
    return ARC_ZE_OK;
}

/**
 * @brief Take the allocation table spin lock.
 *
 * @param ctx Context owning the table.
 */
static void arc_table_lock(ArcZeContext* ctx) {
    while (atomic_flag_test_and_set_explicit(&ctx->table_lock, memory_order_acquire)) {
    }
}

/**
 * @brief Release the allocation table spin lock.
 *
 * @param ctx Context owning the table.
 */
static void arc_table_unlock(ArcZeContext* ctx) {
    atomic_flag_clear_explicit(&ctx->table_lock, memory_order_release);
}

/**
 * @brief Release driver objects in reverse creation order and free the context.
 *
 * @param ctx Context to tear down; may be partially initialized.
 */
static void arc_teardown(ArcZeContext* ctx) {
    for (uint32_t i = 0; i < ctx->alloc_count; ++i) {
        if (ctx->allocs[i].ptr && ctx->api.zeMemFree && ctx->context_handle) {
            ctx->api.zeMemFree(ctx->context_handle, ctx->allocs[i].ptr);
        }
    }
    ctx->alloc_count = 0;
    if (ctx->cmdlist_handle && ctx->api.zeCommandListDestroy) {
        ctx->api.zeCommandListDestroy(ctx->cmdlist_handle);
    }
    if (ctx->queue_handle && ctx->api.zeCommandQueueDestroy) {
        ctx->api.zeCommandQueueDestroy(ctx->queue_handle);
    }
    if (ctx->context_handle && ctx->api.zeContextDestroy) {
        ctx->api.zeContextDestroy(ctx->context_handle);
    }
    if (ctx->ze_module) {
        FREE_LIB(ctx->ze_module);
    }
    memset(&ctx->api, 0, sizeof ctx->api);
    free(ctx);
}

ARC_EXPORT ArcZeContext* arc_ze_init_ex(int32_t* out_status) {
    ArcZeContext* ctx = (ArcZeContext*)calloc(1, sizeof(ArcZeContext));
    if (!ctx) {
        if (out_status) {
            *out_status = ARC_ZE_ERR_LOAD;
        }
        return NULL;
    }
    atomic_init(&ctx->last_status, ARC_ZE_OK);
    atomic_init(&ctx->last_result, 0);
    atomic_init(&ctx->is_dead, false);
    atomic_init(&ctx->pending_cmds, 0);
    atomic_flag_clear(&ctx->table_lock);
    ctx->sync_timeout_ms = arc_configured_sync_timeout_ms();

    int status = ARC_ZE_OK;
    ctx->ze_module = arc_load_loader();
    if (!ctx->ze_module) {
        status = ARC_ZE_ERR_LOAD;
    } else if (!arc_bind_all(ctx)) {
        status = ARC_ZE_ERR_SYMBOL;
    } else {
        ze_result_t res = ctx->api.zeInit(ZE_INIT_FLAG_GPU_ONLY);
        status = res == ZE_RESULT_SUCCESS ? arc_select_gpu(ctx) : arc_ze_api_error(ctx, res);
    }
    if (status == ARC_ZE_OK) {
        status = arc_create_objects(ctx);
    }

    if (status != ARC_ZE_OK) {
        if (out_status) {
            *out_status = status;
        }
        arc_teardown(ctx);
        return NULL;
    }
    if (out_status) {
        *out_status = ARC_ZE_OK;
    }
    return ctx;
}

ARC_EXPORT ArcZeContext* arc_ze_init(void) {
    return arc_ze_init_ex(NULL);
}

/**
 * @brief Record a successful allocation in the table or free it if the table is full.
 *
 * @param ctx Context owning the table.
 * @param ptr Allocation to track.
 * @param size_bytes Allocation size.
 * @param type Residency type.
 * @return true if tracked.
 */
static bool arc_track_alloc(ArcZeContext* ctx, void* ptr, size_t size_bytes, ArcUsmType type) {
    bool tracked = false;
    arc_table_lock(ctx);
    if (ctx->alloc_count < ARC_MAX_TRACKED_ALLOCS) {
        ctx->allocs[ctx->alloc_count].ptr = ptr;
        ctx->allocs[ctx->alloc_count].size = size_bytes;
        ctx->allocs[ctx->alloc_count].type = type;
        ctx->alloc_count++;
        tracked = true;
    }
    arc_table_unlock(ctx);
    return tracked;
}

/**
 * @brief Check that [ptr, ptr + len) lies entirely inside one tracked allocation (overflow-safe).
 *
 * The caller must hold the table lock.
 *
 * @param ctx Context.
 * @param ptr Start address.
 * @param len Length in bytes; must be positive.
 * @param out_type Receives the allocation's residency type when found.
 * @return true if the whole range is inside a single tracked allocation.
 */
static bool arc_range_in_alloc_locked(const ArcZeContext* ctx, const void* ptr, size_t len, ArcUsmType* out_type) {
    const uintptr_t p = (uintptr_t)ptr;
    for (uint32_t i = 0; i < ctx->alloc_count; ++i) {
        const uintptr_t base = (uintptr_t)ctx->allocs[i].ptr;
        const size_t size = ctx->allocs[i].size;
        if (p >= base && (size_t)(p - base) <= size && len <= size - (size_t)(p - base)) {
            *out_type = ctx->allocs[i].type;
            return true;
        }
    }
    return false;
}


ARC_EXPORT void* arc_ze_alloc_usm(ArcZeContext* ctx, size_t size_bytes, ArcUsmType usm_type) {
    if (!ctx || !ctx->context_handle) {
        return NULL;
    }
    if (atomic_load_explicit(&ctx->is_dead, memory_order_acquire)) {
        arc_ze_set_status(ctx, ARC_ZE_ERR_DEAD);
        return NULL;
    }
    if (size_bytes == 0 || (uint64_t)size_bytes > ARC_MAX_USM_BYTES || usm_type > ARC_USM_SHARED) {
        arc_ze_set_status(ctx, size_bytes == 0 || usm_type > ARC_USM_SHARED ? ARC_ZE_ERR_INVALID_ARG : ARC_ZE_ERR_TOO_LARGE);
        return NULL;
    }

    void* ptr = NULL;
    ze_result_t res;
    ze_host_mem_alloc_desc_t host_desc;
    ze_device_mem_alloc_desc_t device_desc;
    memset(&host_desc, 0, sizeof host_desc);
    memset(&device_desc, 0, sizeof device_desc);
    host_desc.stype = ZE_STYPE_HOST_MEM_ALLOC_DESC;
    device_desc.stype = ZE_STYPE_DEVICE_MEM_ALLOC_DESC;

    if (usm_type == ARC_USM_HOST) {
        res = ctx->api.zeMemAllocHost(ctx->context_handle, &host_desc, size_bytes, ARC_USM_ALIGNMENT, &ptr);
    } else if (usm_type == ARC_USM_DEVICE) {
        res = ctx->api.zeMemAllocDevice(ctx->context_handle, &device_desc, size_bytes, ARC_USM_ALIGNMENT, ctx->device_handle, &ptr);
    } else {
        res = ctx->api.zeMemAllocShared(ctx->context_handle, &device_desc, &host_desc, size_bytes, ARC_USM_ALIGNMENT, ctx->device_handle, &ptr);
    }
    if (res != ZE_RESULT_SUCCESS || !ptr) {
        arc_ze_api_error(ctx, res);
        return NULL;
    }
    if (!arc_track_alloc(ctx, ptr, size_bytes, usm_type)) {
        ctx->api.zeMemFree(ctx->context_handle, ptr);
        arc_ze_set_status(ctx, ARC_ZE_ERR_TOO_MANY_ALLOCS);
        return NULL;
    }
    arc_ze_set_status(ctx, ARC_ZE_OK);
    return ptr;
}

ARC_EXPORT int arc_ze_dma_copy_async(ArcZeContext* ctx, void* dst, const void* src, size_t size_bytes) {
    if (!ctx || !ctx->cmdlist_handle || !dst || !src || size_bytes == 0) {
        return arc_ze_set_status(ctx, ARC_ZE_ERR_INVALID_ARG);
    }
    if (atomic_load_explicit(&ctx->is_dead, memory_order_acquire)) {
        return arc_ze_set_status(ctx, ARC_ZE_ERR_DEAD);
    }

    ze_result_t res = ctx->api.zeCommandListAppendMemoryCopy(ctx->cmdlist_handle, dst, src, size_bytes, NULL, 0, NULL);
    if (res == ZE_RESULT_SUCCESS) {
        atomic_fetch_add_explicit(&ctx->pending_cmds, 1, memory_order_acq_rel);
        return arc_ze_set_status(ctx, ARC_ZE_OK);
    }

    /* The CPU fallback runs now, so it is only valid when no GPU command is queued ahead of it
     * (otherwise it would reorder against them) and when both full ranges are CPU-visible
     * allocations; the check and the copy share one critical section so a concurrent free cannot
     * invalidate them in between. */
    if (atomic_load_explicit(&ctx->pending_cmds, memory_order_acquire) == 0) {
        ArcUsmType dst_type = ARC_USM_DEVICE;
        ArcUsmType src_type = ARC_USM_DEVICE;
        arc_table_lock(ctx);
        const bool covered = arc_range_in_alloc_locked(ctx, dst, size_bytes, &dst_type) &&
                             arc_range_in_alloc_locked(ctx, src, size_bytes, &src_type) &&
                             dst_type != ARC_USM_DEVICE && src_type != ARC_USM_DEVICE;
        if (covered) {
            memmove(dst, src, size_bytes);
        }
        arc_table_unlock(ctx);
        if (covered) {
            atomic_store_explicit(&ctx->last_result, res, memory_order_relaxed);
            arc_ze_set_status(ctx, ARC_ZE_OK);
            return ARC_ZE_FALLBACK_MEMCPY;
        }
    }
    return arc_ze_api_error(ctx, res);
}

ARC_EXPORT int arc_ze_synchronize(ArcZeContext* ctx) {
    if (!ctx || !ctx->queue_handle || !ctx->cmdlist_handle) {
        return arc_ze_set_status(ctx, ARC_ZE_ERR_INVALID_ARG);
    }
    if (atomic_load_explicit(&ctx->is_dead, memory_order_acquire)) {
        return arc_ze_set_status(ctx, ARC_ZE_ERR_DEAD);
    }

    ze_result_t res = ctx->api.zeCommandListClose(ctx->cmdlist_handle);
    if (res != ZE_RESULT_SUCCESS) {
        return arc_ze_api_error(ctx, res);
    }
    res = ctx->api.zeCommandQueueExecuteCommandLists(ctx->queue_handle, 1, &ctx->cmdlist_handle, NULL);
    if (res != ZE_RESULT_SUCCESS) {
        atomic_store_explicit(&ctx->is_dead, res == ZE_RESULT_ERROR_DEVICE_LOST, memory_order_release);
        return arc_ze_api_error(ctx, res);
    }

    const uint64_t deadline = arc_now_ms() + ctx->sync_timeout_ms;
    for (;;) {
        const uint64_t now = arc_now_ms();
        const uint64_t remaining = deadline > now ? deadline - now : 0;
        const uint64_t slice_ms = remaining < ARC_SYNC_SLICE_MS ? remaining : ARC_SYNC_SLICE_MS;
        res = ctx->api.zeCommandQueueSynchronize(ctx->queue_handle, slice_ms * 1000000u);
        if (res == ZE_RESULT_SUCCESS) {
            break;
        }
        if (res != ZE_RESULT_NOT_READY) {
            atomic_store_explicit(&ctx->is_dead, true, memory_order_release);
            return arc_ze_api_error(ctx, res);
        }
        if (arc_now_ms() >= deadline) {
            atomic_store_explicit(&ctx->is_dead, true, memory_order_release);
            return arc_ze_set_status(ctx, ARC_ZE_ERR_TIMEOUT);
        }
    }

    atomic_store_explicit(&ctx->pending_cmds, 0, memory_order_release);
    res = ctx->api.zeCommandListReset(ctx->cmdlist_handle);
    return res == ZE_RESULT_SUCCESS ? arc_ze_set_status(ctx, ARC_ZE_OK) : arc_ze_api_error(ctx, res);
}

ARC_EXPORT void arc_ze_free_usm(ArcZeContext* ctx, void* ptr) {
    if (!ctx || !ctx->context_handle || !ptr) {
        return;
    }

    bool owned = false;
    arc_table_lock(ctx);
    for (uint32_t i = 0; i < ctx->alloc_count; ++i) {
        if (ctx->allocs[i].ptr == ptr) {
            ctx->allocs[i] = ctx->allocs[ctx->alloc_count - 1];
            ctx->alloc_count--;
            owned = true;
            break;
        }
    }
    arc_table_unlock(ctx);

    if (owned) {
        ctx->api.zeMemFree(ctx->context_handle, ptr);
    } else {
        arc_ze_set_status(ctx, ARC_ZE_ERR_INVALID_ARG);
    }
}

ARC_EXPORT void arc_ze_destroy(ArcZeContext* ctx) {
    if (ctx) {
        arc_teardown(ctx);
    }
}

ARC_EXPORT int arc_ze_is_dead(const ArcZeContext* ctx) {
    return ctx && atomic_load_explicit(&ctx->is_dead, memory_order_acquire) ? 1 : 0;
}

ARC_EXPORT int32_t arc_ze_last_status(const ArcZeContext* ctx) {
    return ctx ? (int32_t)atomic_load_explicit(&ctx->last_status, memory_order_relaxed) : ARC_ZE_ERR_NOT_INITIALIZED;
}

ARC_EXPORT uint32_t arc_ze_last_result(const ArcZeContext* ctx) {
    return ctx ? (uint32_t)atomic_load_explicit(&ctx->last_result, memory_order_relaxed) : 0u;
}

ARC_EXPORT uint32_t arc_ze_device_type(const ArcZeContext* ctx) {
    return ctx ? ctx->device_type : 0u;
}

ARC_EXPORT uint32_t arc_ze_alloc_count(const ArcZeContext* ctx) {
    return ctx ? ctx->alloc_count : 0u;
}

ARC_EXPORT uint32_t arc_ze_sync_timeout_ms(const ArcZeContext* ctx) {
    return ctx ? (uint32_t)ctx->sync_timeout_ms : 0u;
}
