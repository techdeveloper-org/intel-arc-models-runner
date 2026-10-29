/**
 * @file arc_flash_stream_mmap.c
 * @brief Implementation of bare-metal Win32 memory-mapped model streaming.
 */

#include "arc_flash_stream_mmap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

/**
 * @brief Open and memory-map a model container file using Win32 API.
 *
 * @param file_path Filesystem path to the model container.
 * @return ArcMmapContext* Initialized context pointer, or NULL on failure.
 */
ARC_EXPORT ArcMmapContext* arc_mmap_open(const char* file_path) {
    if (!file_path || strlen(file_path) == 0) {
        return NULL;
    }

    ArcMmapContext* ctx = (ArcMmapContext*)calloc(1, sizeof(ArcMmapContext));
    if (!ctx) {
        return NULL;
    }

#if defined(_WIN32)
    HANDLE hFile = CreateFileA(
        file_path,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        free(ctx);
        return NULL;
    }

    LARGE_INTEGER liSize;
    if (!GetFileSizeEx(hFile, &liSize) || liSize.QuadPart <= 0) {
        CloseHandle(hFile);
        free(ctx);
        return NULL;
    }

    HANDLE hMap = CreateFileMappingA(
        hFile,
        NULL,
        PAGE_READONLY,
        liSize.HighPart,
        liSize.LowPart,
        NULL
    );

    if (!hMap) {
        CloseHandle(hFile);
        free(ctx);
        return NULL;
    }

    void* base_ptr = MapViewOfFile(
        hMap,
        FILE_MAP_READ,
        0,
        0,
        0
    );

    if (!base_ptr) {
        CloseHandle(hMap);
        CloseHandle(hFile);
        free(ctx);
        return NULL;
    }

    ctx->file_handle = (void*)hFile;
    ctx->map_handle = (void*)hMap;
    ctx->base_ptr = (uint8_t*)base_ptr;
    ctx->file_size = (uint64_t)liSize.QuadPart;
    ctx->total_layers = 0;
    ctx->is_locked = false;
#else
    int fd = open(file_path, O_RDONLY);
    if (fd < 0) {
        free(ctx);
        return NULL;
    }

    struct stat sb;
    if (fstat(fd, &sb) < 0 || sb.st_size <= 0) {
        close(fd);
        free(ctx);
        return NULL;
    }

    void* base_ptr = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (base_ptr == MAP_FAILED) {
        close(fd);
        free(ctx);
        return NULL;
    }

    ctx->file_handle = (void*)(intptr_t)fd;
    ctx->map_handle = NULL;
    ctx->base_ptr = (uint8_t*)base_ptr;
    ctx->file_size = (uint64_t)sb.st_size;
    ctx->total_layers = 0;
    ctx->is_locked = false;
#endif

    return ctx;
}

/**
 * @brief Register a layer descriptor into the context's binary lookup table.
 *
 * @param ctx Context handle returned by arc_mmap_open.
 * @param layer Pointer to the layer descriptor to register.
 * @return int 0 on success, non-zero error code on failure.
 */
ARC_EXPORT int arc_mmap_register_layer(ArcMmapContext* ctx, const ArcLayerDescriptor* layer) {
    if (!ctx || !layer) {
        return -1;
    }

    if (ctx->total_layers >= ARC_MAX_LAYERS) {
        return -2;
    }

    for (uint32_t i = 0; i < ctx->total_layers; ++i) {
        if (ctx->layers[i].layer_idx == layer->layer_idx) {
            memcpy(&ctx->layers[i], layer, sizeof(ArcLayerDescriptor));
            return 0;
        }
    }

    memcpy(&ctx->layers[ctx->total_layers], layer, sizeof(ArcLayerDescriptor));
    ctx->total_layers++;
    return 0;
}

/**
 * @brief Retrieve a direct memory pointer to a specific tensor payload.
 *
 * @param ctx Context handle returned by arc_mmap_open.
 * @param layer_idx Zero-based layer index.
 * @param tensor_name Short name of the tensor within the layer.
 * @param out_size Pointer to receive the byte size of the tensor payload.
 * @return const uint8_t* Direct pointer to the mapped memory, or NULL if not found.
 */
ARC_EXPORT const uint8_t* arc_mmap_get_tensor_ptr(
    ArcMmapContext* ctx,
    int32_t layer_idx,
    const char* tensor_name,
    uint64_t* out_size
) {
    if (!ctx || !ctx->base_ptr || !tensor_name) {
        return NULL;
    }

    for (uint32_t i = 0; i < ctx->total_layers; ++i) {
        if (ctx->layers[i].layer_idx == layer_idx) {
            const ArcLayerDescriptor* lyr = &ctx->layers[i];
            for (uint32_t j = 0; j < lyr->tensor_count; ++j) {
                if (strncmp(lyr->tensors[j].name, tensor_name, ARC_MAX_NAME_LEN) == 0) {
                    uint64_t offset = lyr->tensors[j].file_offset;
                    uint64_t size = lyr->tensors[j].size_bytes;

                    if (offset + size > ctx->file_size) {
                        return NULL;
                    }

                    if (out_size) {
                        *out_size = size;
                    }
                    return ctx->base_ptr + offset;
                }
            }
            return NULL;
        }
    }

    return NULL;
}

/**
 * @brief Lock a memory range into physical RAM to prevent OS paging thrash.
 *
 * @param ctx Context handle returned by arc_mmap_open.
 * @param offset Byte offset from the base mapping pointer.
 * @param size Number of bytes to lock into physical memory.
 * @return int 0 on success, non-zero if VirtualLock fails.
 */
ARC_EXPORT int arc_mmap_lock_range(ArcMmapContext* ctx, uint64_t offset, uint64_t size) {
    if (!ctx || !ctx->base_ptr || offset + size > ctx->file_size) {
        return -1;
    }

#if defined(_WIN32)
    BOOL ok = VirtualLock((LPVOID)(ctx->base_ptr + offset), (SIZE_T)size);
    if (!ok) {
        return (int)GetLastError();
    }
    ctx->is_locked = true;
    return 0;
#else
    int res = mlock((const void*)(ctx->base_ptr + offset), (size_t)size);
    if (res != 0) {
        return res;
    }
    ctx->is_locked = true;
    return 0;
#endif
}

/**
 * @brief Unlock a previously locked physical memory range.
 *
 * @param ctx Context handle returned by arc_mmap_open.
 * @param offset Byte offset from the base mapping pointer.
 * @param size Number of bytes to unlock.
 * @return int 0 on success, non-zero on error.
 */
ARC_EXPORT int arc_mmap_unlock_range(ArcMmapContext* ctx, uint64_t offset, uint64_t size) {
    if (!ctx || !ctx->base_ptr || offset + size > ctx->file_size) {
        return -1;
    }

#if defined(_WIN32)
    BOOL ok = VirtualUnlock((LPVOID)(ctx->base_ptr + offset), (SIZE_T)size);
    if (!ok) {
        return (int)GetLastError();
    }
    ctx->is_locked = false;
    return 0;
#else
    int res = munlock((const void*)(ctx->base_ptr + offset), (size_t)size);
    if (res != 0) {
        return res;
    }
    ctx->is_locked = false;
    return 0;
#endif
}

/**
 * @brief Unmap views and close all operating system file handles.
 *
 * @param ctx Context handle to close and deallocate.
 */
ARC_EXPORT void arc_mmap_close(ArcMmapContext* ctx) {
    if (!ctx) {
        return;
    }

#if defined(_WIN32)
    if (ctx->base_ptr) {
        UnmapViewOfFile((LPCVOID)ctx->base_ptr);
        ctx->base_ptr = NULL;
    }
    if (ctx->map_handle) {
        CloseHandle((HANDLE)ctx->map_handle);
        ctx->map_handle = NULL;
    }
    if (ctx->file_handle && ctx->file_handle != INVALID_HANDLE_VALUE) {
        CloseHandle((HANDLE)ctx->file_handle);
        ctx->file_handle = NULL;
    }
#else
    if (ctx->base_ptr) {
        munmap(ctx->base_ptr, (size_t)ctx->file_size);
        ctx->base_ptr = NULL;
    }
    if (ctx->file_handle) {
        close((int)(intptr_t)ctx->file_handle);
        ctx->file_handle = NULL;
    }
#endif

    free(ctx);
}
