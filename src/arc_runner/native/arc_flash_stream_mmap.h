/**
 * @file arc_flash_stream_mmap.h
 * @brief Bounds-checked memory-mapped model streaming interface (Win32 and POSIX).
 *
 * Provides zero-copy access to memory-mapped container files with best-effort
 * physical memory locking and a pre-indexed layer table. Every extent supplied by
 * the caller is validated against the mapped file size before a pointer is derived
 * from it (ISO C17 6.5.6p8: pointer arithmetic must stay within the object or one
 * past its end).
 *
 * Threading contract: arc_mmap_register_layer and arc_mmap_seal must be called from
 * one thread. After arc_mmap_seal returns, the layer table is immutable and
 * arc_mmap_get_tensor_ptr may be called concurrently from any thread.
 */

#ifndef ARC_FLASH_STREAM_MMAP_H
#define ARC_FLASH_STREAM_MMAP_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(_WIN32)
  #define ARC_EXPORT __declspec(dllexport)
#else
  #define ARC_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ARC_MAX_PATH 1024
#define ARC_MAX_TENSORS_PER_LAYER 64
#define ARC_MAX_LAYERS 256
#define ARC_MAX_NAME_LEN 128
#define ARC_NATIVE_ABI_VERSION 3u

/**
 * @struct ArcTensorDescriptor
 * @brief Binary descriptor holding pre-calculated file offsets and dimensions for a tensor.
 */
typedef struct {
    char name[ARC_MAX_NAME_LEN];
    uint32_t tensor_type;
    uint32_t n_dims;
    uint64_t dims[4];
    uint64_t file_offset;
    uint64_t size_bytes;
} ArcTensorDescriptor;

/**
 * @struct ArcLayerDescriptor
 * @brief Binary descriptor mapping a layer index to its constituent tensors.
 */
typedef struct {
    int32_t layer_idx;
    uint32_t tensor_count;
    uint64_t total_layer_bytes;
    ArcTensorDescriptor tensors[ARC_MAX_TENSORS_PER_LAYER];
} ArcLayerDescriptor;

_Static_assert(sizeof(ArcTensorDescriptor) == 184, "ArcTensorDescriptor layout is part of the ABI");
_Static_assert(sizeof(ArcLayerDescriptor) == 11792, "ArcLayerDescriptor layout is part of the ABI");

/**
 * @enum ArcStatus
 * @brief Result codes returned by the arc_mmap_* functions.
 */
typedef enum ArcStatus {
    ARC_OK = 0,
    ARC_ERR_INVALID_ARG = -1,
    ARC_ERR_RANGE = -2,
    ARC_ERR_TOO_MANY_TENSORS = -3,
    ARC_ERR_BAD_EXTENT = -4,
    ARC_ERR_SEALED = -5,
    ARC_ERR_NOT_SEALED = -6,
    ARC_ERR_LOCK_FAILED = -7,
    ARC_ERR_OPEN_FAILED = -8,
    ARC_ERR_CLOSED = -9,
    ARC_ERR_NOT_FOUND = -10
} ArcStatus;

/**
 * @struct ArcMmapContext
 * @brief Opaque context managing the mapped file handles and the layer table.
 */
typedef struct ArcMmapContext ArcMmapContext;

/**
 * @brief Check that the byte range [off, off + sz) lies inside a file of fsize bytes.
 *
 * Written as two comparisons that cannot overflow: sz <= fsize guarantees the
 * subtraction fsize - sz is non-negative (unsigned arithmetic is defined modulo
 * 2^N, C17 6.2.5p9, but must not be relied on for bounds).
 *
 * @param fsize Size of the mapped file in bytes.
 * @param off Start of the range.
 * @param sz Length of the range.
 * @return true if the whole range is inside the file.
 */
static inline bool arc_range_ok(uint64_t fsize, uint64_t off, uint64_t sz) {
    return sz <= fsize && off <= fsize - sz;
}

/**
 * @brief Open and memory-map a model container file read-only.
 *
 * @param utf8_path Filesystem path encoded as UTF-8.
 * @param out_status Optional receiver for an ArcStatus code (ARC_OK on success).
 * @return Initialized context pointer, or NULL on failure.
 */
ARC_EXPORT ArcMmapContext* arc_mmap_open(const char* utf8_path, int32_t* out_status);

/**
 * @brief Validate and register a layer descriptor into the lookup table.
 *
 * The descriptor is copied, validated (layer index, tensor count, dimension count,
 * every extent against the file size), and published only if all checks pass.
 * Registering an existing layer index replaces it until the table is sealed.
 *
 * @param ctx Context handle returned by arc_mmap_open.
 * @param layer Pointer to the layer descriptor to register.
 * @return ARC_OK on success, otherwise a negative ArcStatus code.
 */
ARC_EXPORT int arc_mmap_register_layer(ArcMmapContext* ctx, const ArcLayerDescriptor* layer);

/**
 * @brief Retrieve a direct memory pointer to a specific tensor payload.
 *
 * Requires arc_mmap_seal to have been called. On failure returns NULL and records
 * the reason, readable through arc_mmap_last_status.
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
);

/**
 * @brief Best-effort lock of a mapped range into physical RAM.
 *
 * @param ctx Context handle returned by arc_mmap_open.
 * @param offset Byte offset from the base mapping pointer.
 * @param size Number of bytes to lock; must be non-zero.
 * @return ARC_OK on success, ARC_ERR_LOCK_FAILED if the OS refused (the OS error is
 *         available from arc_mmap_last_os_error), or another negative ArcStatus.
 */
ARC_EXPORT int arc_mmap_lock_range(ArcMmapContext* ctx, uint64_t offset, uint64_t size);

/**
 * @brief Unlock a previously locked physical memory range.
 *
 * @param ctx Context handle returned by arc_mmap_open.
 * @param offset Byte offset from the base mapping pointer.
 * @param size Number of bytes to unlock.
 * @return ARC_OK on success, otherwise a negative ArcStatus code.
 */
ARC_EXPORT int arc_mmap_unlock_range(ArcMmapContext* ctx, uint64_t offset, uint64_t size);

/**
 * @brief Unmap views and close all operating system file handles.
 *
 * @param ctx Context handle to close and deallocate.
 */
ARC_EXPORT void arc_mmap_close(ArcMmapContext* ctx);

/**
 * @brief Freeze the layer table so lookups can run concurrently without locks.
 *
 * @param ctx Context handle returned by arc_mmap_open.
 * @return ARC_OK (idempotent), or ARC_ERR_INVALID_ARG for a NULL context.
 */
ARC_EXPORT int arc_mmap_seal(ArcMmapContext* ctx);

/** @brief Return 1 if the layer table is sealed, 0 otherwise. */
ARC_EXPORT int arc_mmap_is_sealed(const ArcMmapContext* ctx);

/** @brief Return the last ArcStatus recorded on the context (diagnostic, relaxed). */
ARC_EXPORT int32_t arc_mmap_last_status(const ArcMmapContext* ctx);

/** @brief Return the last operating-system error code recorded on the context. */
ARC_EXPORT uint32_t arc_mmap_last_os_error(const ArcMmapContext* ctx);

/** @brief Return the mapped file size in bytes, or 0 for a NULL context. */
ARC_EXPORT uint64_t arc_mmap_file_size(const ArcMmapContext* ctx);

/** @brief Return the number of registered layers. */
ARC_EXPORT uint32_t arc_mmap_layer_count(const ArcMmapContext* ctx);

/** @brief Return the number of bytes currently locked through arc_mmap_lock_range. */
ARC_EXPORT uint64_t arc_mmap_locked_bytes(const ArcMmapContext* ctx);

/** @brief Return ARC_NATIVE_ABI_VERSION; the Python bridge refuses other values. */
ARC_EXPORT uint32_t arc_native_abi_version(void);

/** @brief Return a static string describing platform and compiler of this build. */
ARC_EXPORT const char* arc_native_build_info(void);

#ifdef __cplusplus
}
#endif

#endif /* ARC_FLASH_STREAM_MMAP_H */
