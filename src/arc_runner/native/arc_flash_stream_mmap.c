/**
 * @file arc_flash_stream_mmap.c
 * @brief Bounds-checked memory-mapped model streaming (Win32 and POSIX).
 */

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "arc_flash_stream_mmap.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
  #include <windows.h>
#else
  #include <errno.h>
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

/**
 * @struct ArcMmapContext
 * @brief Mapped file state; the layer table is immutable once sealed is set.
 */
struct ArcMmapContext {
    void* file_handle;
    void* map_handle;
    uint8_t* base_ptr;
    uint64_t file_size;
    uint32_t total_layers;
    ArcLayerDescriptor layers[ARC_MAX_LAYERS];
    atomic_uint_fast64_t locked_bytes;
    atomic_bool sealed;
    atomic_int_fast32_t last_status;
    atomic_uint_fast32_t last_os_error;
};

/**
 * @brief Record a status code (diagnostic only, relaxed ordering) and return it.
 *
 * @param ctx Context to annotate; ignored when NULL.
 * @param status ArcStatus value to store.
 * @return The same status, so callers can write "return arc_set_status(...)".
 */
static int arc_set_status(ArcMmapContext* ctx, int status) {
    if (ctx) {
        atomic_store_explicit(&ctx->last_status, status, memory_order_relaxed);
    }
    return status;
}

/**
 * @brief Record the operating-system error that caused a failure.
 *
 * @param ctx Context to annotate; ignored when NULL.
 * @param code Win32 GetLastError value or POSIX errno.
 */
static void arc_set_os_error(ArcMmapContext* ctx, uint32_t code) {
    if (ctx) {
        atomic_store_explicit(&ctx->last_os_error, code, memory_order_relaxed);
    }
}

/**
 * @brief Report an open failure through the optional out parameter and return NULL.
 *
 * @param out_status Optional receiver for the status code.
 * @param status ArcStatus value describing the failure.
 * @return Always NULL.
 */
static ArcMmapContext* arc_open_fail(int32_t* out_status, int status) {
    if (out_status) {
        *out_status = status;
    }
    return NULL;
}

/**
 * @brief Check that a 64-bit size or offset is representable as size_t on this target.
 *
 * @param value Value to test.
 * @return true if value fits in size_t (always true on 64-bit targets).
 */
static bool arc_fits_size_t(uint64_t value) {
#if SIZE_MAX < UINT64_MAX
    return value <= (uint64_t)SIZE_MAX;
#else
    (void)value;
    return true;
#endif
}

#if defined(_WIN32)
/**
 * @brief Convert a UTF-8 path to a wide path, adding the \\?\ prefix for long drive paths.
 *
 * @param utf8 NUL-terminated UTF-8 path.
 * @return Heap-allocated wide string (free with free), or NULL on conversion failure.
 */
static wchar_t* arc_utf8_to_wide_path(const char* utf8) {
    int needed = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, NULL, 0);
    if (needed <= 0) {
        return NULL;
    }

    bool is_drive_path = ((utf8[0] >= 'A' && utf8[0] <= 'Z') || (utf8[0] >= 'a' && utf8[0] <= 'z'))
                         && utf8[1] == ':' && (utf8[2] == '\\' || utf8[2] == '/');
    bool add_prefix = is_drive_path && needed >= MAX_PATH;
    size_t prefix_len = add_prefix ? 4u : 0u;

    wchar_t* wide = (wchar_t*)malloc(((size_t)needed + prefix_len) * sizeof(wchar_t));
    if (!wide) {
        return NULL;
    }
    if (add_prefix) {
        wide[0] = L'\\';
        wide[1] = L'\\';
        wide[2] = L'?';
        wide[3] = L'\\';
    }
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, wide + prefix_len, needed) <= 0) {
        free(wide);
        return NULL;
    }
    if (add_prefix) {
        for (wchar_t* c = wide + prefix_len; *c; ++c) {
            if (*c == L'/') {
                *c = L'\\';
            }
        }
    }
    return wide;
}
#endif

ARC_EXPORT ArcMmapContext* arc_mmap_open(const char* utf8_path, int32_t* out_status) {
    if (out_status) {
        *out_status = ARC_OK;
    }
    if (!utf8_path || utf8_path[0] == '\0') {
        return arc_open_fail(out_status, ARC_ERR_INVALID_ARG);
    }

    ArcMmapContext* ctx = (ArcMmapContext*)calloc(1, sizeof(ArcMmapContext));
    if (!ctx) {
        return arc_open_fail(out_status, ARC_ERR_OPEN_FAILED);
    }

#if defined(_WIN32)
    wchar_t* wide_path = arc_utf8_to_wide_path(utf8_path);
    if (!wide_path) {
        free(ctx);
        return arc_open_fail(out_status, ARC_ERR_INVALID_ARG);
    }

    HANDLE hFile = CreateFileW(
        wide_path,
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_DELETE,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
        NULL
    );
    free(wide_path);
    if (hFile == INVALID_HANDLE_VALUE) {
        free(ctx);
        return arc_open_fail(out_status, ARC_ERR_OPEN_FAILED);
    }

    LARGE_INTEGER liSize;
    if (!GetFileSizeEx(hFile, &liSize) || liSize.QuadPart <= 0 || !arc_fits_size_t((uint64_t)liSize.QuadPart)) {
        CloseHandle(hFile);
        free(ctx);
        return arc_open_fail(out_status, ARC_ERR_OPEN_FAILED);
    }

    HANDLE hMap = CreateFileMappingW(hFile, NULL, PAGE_READONLY, (DWORD)liSize.HighPart, (DWORD)liSize.LowPart, NULL);
    if (!hMap) {
        CloseHandle(hFile);
        free(ctx);
        return arc_open_fail(out_status, ARC_ERR_OPEN_FAILED);
    }

    void* base_ptr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!base_ptr) {
        CloseHandle(hMap);
        CloseHandle(hFile);
        free(ctx);
        return arc_open_fail(out_status, ARC_ERR_OPEN_FAILED);
    }

    ctx->file_handle = (void*)hFile;
    ctx->map_handle = (void*)hMap;
    ctx->base_ptr = (uint8_t*)base_ptr;
    ctx->file_size = (uint64_t)liSize.QuadPart;
#else
    int fd = open(utf8_path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        free(ctx);
        return arc_open_fail(out_status, ARC_ERR_OPEN_FAILED);
    }

    struct stat sb;
    if (fstat(fd, &sb) < 0 || sb.st_size <= 0 || !arc_fits_size_t((uint64_t)sb.st_size)) {
        close(fd);
        free(ctx);
        return arc_open_fail(out_status, ARC_ERR_OPEN_FAILED);
    }

    void* base_ptr = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (base_ptr == MAP_FAILED) {
        close(fd);
        free(ctx);
        return arc_open_fail(out_status, ARC_ERR_OPEN_FAILED);
    }

    ctx->file_handle = (void*)((intptr_t)fd + 1); /* fd + 1 so that descriptor 0 is distinguishable from "no file" */
    ctx->map_handle = NULL;
    ctx->base_ptr = (uint8_t*)base_ptr;
    ctx->file_size = (uint64_t)sb.st_size;
#endif

    atomic_init(&ctx->locked_bytes, 0);
    atomic_init(&ctx->sealed, false);
    atomic_init(&ctx->last_status, ARC_OK);
    atomic_init(&ctx->last_os_error, 0);
    return ctx;
}

/**
 * @brief Validate a layer descriptor copy against the mapped file and normalize it.
 *
 * @param file_size Size of the mapped file in bytes.
 * @param local Private copy of the caller's descriptor; names are NUL-terminated and
 *              total_layer_bytes is recomputed on success.
 * @return ARC_OK, or the ArcStatus describing the first violated rule.
 */
static int arc_validate_layer(uint64_t file_size, ArcLayerDescriptor* local) {
    if (local->layer_idx < 0 || local->layer_idx >= ARC_MAX_LAYERS) {
        return ARC_ERR_INVALID_ARG;
    }
    if (local->tensor_count > ARC_MAX_TENSORS_PER_LAYER) {
        return ARC_ERR_TOO_MANY_TENSORS;
    }

    uint64_t total = 0;
    for (uint32_t i = 0; i < local->tensor_count; ++i) {
        ArcTensorDescriptor* t = &local->tensors[i];
        t->name[ARC_MAX_NAME_LEN - 1] = '\0';
        if (t->n_dims > 4) {
            return ARC_ERR_INVALID_ARG;
        }
        if (!arc_range_ok(file_size, t->file_offset, t->size_bytes)) {
            return ARC_ERR_BAD_EXTENT;
        }
        if (t->size_bytes > UINT64_MAX - total) {
            return ARC_ERR_BAD_EXTENT;
        }
        total += t->size_bytes;
    }
    local->total_layer_bytes = total;
    return ARC_OK;
}

ARC_EXPORT int arc_mmap_register_layer(ArcMmapContext* ctx, const ArcLayerDescriptor* layer) {
    if (!ctx || !layer) {
        return arc_set_status(ctx, ARC_ERR_INVALID_ARG);
    }
    if (atomic_load_explicit(&ctx->sealed, memory_order_acquire)) {
        return arc_set_status(ctx, ARC_ERR_SEALED);
    }

    ArcLayerDescriptor local;
    memcpy(&local, layer, sizeof(ArcLayerDescriptor));
    int status = arc_validate_layer(ctx->file_size, &local);
    if (status != ARC_OK) {
        return arc_set_status(ctx, status);
    }

    for (uint32_t i = 0; i < ctx->total_layers; ++i) {
        if (ctx->layers[i].layer_idx == local.layer_idx) {
            memcpy(&ctx->layers[i], &local, sizeof(ArcLayerDescriptor));
            return arc_set_status(ctx, ARC_OK);
        }
    }
    if (ctx->total_layers >= ARC_MAX_LAYERS) {
        return arc_set_status(ctx, ARC_ERR_RANGE);
    }
    memcpy(&ctx->layers[ctx->total_layers], &local, sizeof(ArcLayerDescriptor));
    ctx->total_layers++;
    return arc_set_status(ctx, ARC_OK);
}

ARC_EXPORT int arc_mmap_seal(ArcMmapContext* ctx) {
    if (!ctx) {
        return ARC_ERR_INVALID_ARG;
    }
    atomic_store_explicit(&ctx->sealed, true, memory_order_release);
    return arc_set_status(ctx, ARC_OK);
}

ARC_EXPORT const uint8_t* arc_mmap_get_tensor_ptr(
    ArcMmapContext* ctx,
    int32_t layer_idx,
    const char* tensor_name,
    uint64_t* out_size
) {
    if (!ctx || !ctx->base_ptr || !tensor_name) {
        arc_set_status(ctx, ARC_ERR_INVALID_ARG);
        return NULL;
    }
    if (!atomic_load_explicit(&ctx->sealed, memory_order_acquire)) {
        arc_set_status(ctx, ARC_ERR_NOT_SEALED);
        return NULL;
    }

    for (uint32_t i = 0; i < ctx->total_layers; ++i) {
        const ArcLayerDescriptor* lyr = &ctx->layers[i];
        if (lyr->layer_idx != layer_idx) {
            continue;
        }
        for (uint32_t j = 0; j < lyr->tensor_count; ++j) {
            const ArcTensorDescriptor* t = &lyr->tensors[j];
            if (strncmp(t->name, tensor_name, ARC_MAX_NAME_LEN) != 0) {
                continue;
            }
            if (!arc_range_ok(ctx->file_size, t->file_offset, t->size_bytes) || !arc_fits_size_t(t->file_offset)) {
                arc_set_status(ctx, ARC_ERR_BAD_EXTENT);
                return NULL;
            }
            if (out_size) {
                *out_size = t->size_bytes;
            }
            arc_set_status(ctx, ARC_OK);
            return ctx->base_ptr + (size_t)t->file_offset;
        }
        break;
    }

    arc_set_status(ctx, ARC_ERR_NOT_FOUND);
    return NULL;
}

/**
 * @brief Validate a lock or unlock request against the mapping.
 *
 * @param ctx Context handle.
 * @param offset Byte offset into the mapping.
 * @param size Byte count.
 * @return ARC_OK, or the ArcStatus describing the violated rule.
 */
static int arc_check_lock_args(ArcMmapContext* ctx, uint64_t offset, uint64_t size) {
    if (!ctx || !ctx->base_ptr || size == 0) {
        return ARC_ERR_INVALID_ARG;
    }
    if (!arc_range_ok(ctx->file_size, offset, size) || !arc_fits_size_t(offset) || !arc_fits_size_t(size)) {
        return ARC_ERR_RANGE;
    }
    return ARC_OK;
}

/**
 * @brief Subtract from the locked-byte counter without going below zero.
 *
 * @param counter Atomic counter to decrement.
 * @param amount Bytes to subtract; clamped to the current value.
 */
static void arc_saturating_sub(atomic_uint_fast64_t* counter, uint64_t amount) {
    uint64_t current = atomic_load_explicit(counter, memory_order_relaxed);
    uint64_t next;
    do {
        next = amount >= current ? 0 : current - amount;
    } while (!atomic_compare_exchange_weak_explicit(counter, &current, next, memory_order_relaxed, memory_order_relaxed));
}

ARC_EXPORT int arc_mmap_lock_range(ArcMmapContext* ctx, uint64_t offset, uint64_t size) {
    int status = arc_check_lock_args(ctx, offset, size);
    if (status != ARC_OK) {
        return arc_set_status(ctx, status);
    }

#if defined(_WIN32)
    if (!VirtualLock((LPVOID)(ctx->base_ptr + (size_t)offset), (SIZE_T)size)) {
        arc_set_os_error(ctx, (uint32_t)GetLastError());
        return arc_set_status(ctx, ARC_ERR_LOCK_FAILED);
    }
#else
    if (mlock((const void*)(ctx->base_ptr + (size_t)offset), (size_t)size) != 0) {
        arc_set_os_error(ctx, (uint32_t)errno);
        return arc_set_status(ctx, ARC_ERR_LOCK_FAILED);
    }
#endif
    atomic_fetch_add_explicit(&ctx->locked_bytes, size, memory_order_relaxed);
    return arc_set_status(ctx, ARC_OK);
}

ARC_EXPORT int arc_mmap_unlock_range(ArcMmapContext* ctx, uint64_t offset, uint64_t size) {
    int status = arc_check_lock_args(ctx, offset, size);
    if (status != ARC_OK) {
        return arc_set_status(ctx, status);
    }

#if defined(_WIN32)
    if (!VirtualUnlock((LPVOID)(ctx->base_ptr + (size_t)offset), (SIZE_T)size)) {
        arc_set_os_error(ctx, (uint32_t)GetLastError());
        return arc_set_status(ctx, ARC_ERR_LOCK_FAILED);
    }
#else
    if (munlock((const void*)(ctx->base_ptr + (size_t)offset), (size_t)size) != 0) {
        arc_set_os_error(ctx, (uint32_t)errno);
        return arc_set_status(ctx, ARC_ERR_LOCK_FAILED);
    }
#endif
    arc_saturating_sub(&ctx->locked_bytes, size);
    return arc_set_status(ctx, ARC_OK);
}

ARC_EXPORT void arc_mmap_close(ArcMmapContext* ctx) {
    if (!ctx) {
        return;
    }

#if defined(_WIN32)
    if (ctx->base_ptr) {
        UnmapViewOfFile((LPCVOID)ctx->base_ptr);
    }
    if (ctx->map_handle) {
        CloseHandle((HANDLE)ctx->map_handle);
    }
    if (ctx->file_handle && ctx->file_handle != INVALID_HANDLE_VALUE) {
        CloseHandle((HANDLE)ctx->file_handle);
    }
#else
    if (ctx->base_ptr) {
        munmap(ctx->base_ptr, (size_t)ctx->file_size);
    }
    if (ctx->file_handle) {
        close((int)((intptr_t)ctx->file_handle - 1));
    }
#endif

    free(ctx);
}

ARC_EXPORT int arc_mmap_is_sealed(const ArcMmapContext* ctx) {
    return ctx && atomic_load_explicit(&ctx->sealed, memory_order_acquire) ? 1 : 0;
}

ARC_EXPORT int32_t arc_mmap_last_status(const ArcMmapContext* ctx) {
    return ctx ? (int32_t)atomic_load_explicit(&ctx->last_status, memory_order_relaxed) : ARC_ERR_INVALID_ARG;
}

ARC_EXPORT uint32_t arc_mmap_last_os_error(const ArcMmapContext* ctx) {
    return ctx ? (uint32_t)atomic_load_explicit(&ctx->last_os_error, memory_order_relaxed) : 0u;
}

ARC_EXPORT uint64_t arc_mmap_file_size(const ArcMmapContext* ctx) {
    return ctx ? ctx->file_size : 0u;
}

ARC_EXPORT uint32_t arc_mmap_layer_count(const ArcMmapContext* ctx) {
    return ctx ? ctx->total_layers : 0u;
}

ARC_EXPORT uint64_t arc_mmap_locked_bytes(const ArcMmapContext* ctx) {
    return ctx ? (uint64_t)atomic_load_explicit(&ctx->locked_bytes, memory_order_relaxed) : 0u;
}

ARC_EXPORT uint32_t arc_native_abi_version(void) {
    return ARC_NATIVE_ABI_VERSION;
}

ARC_EXPORT const char* arc_native_build_info(void) {
#if defined(_WIN32)
    #define ARC_BUILD_PLATFORM "windows"
#else
    #define ARC_BUILD_PLATFORM "posix"
#endif
    return "arc-native abi=3 platform=" ARC_BUILD_PLATFORM " compiler=" __VERSION__;
}
