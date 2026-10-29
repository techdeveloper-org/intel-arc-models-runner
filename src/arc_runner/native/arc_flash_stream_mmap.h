/**
 * @file arc_flash_stream_mmap.h
 * @brief High-performance bare-metal Win32 memory-mapped model streaming interface.
 *
 * Provides direct zero-copy access to memory-mapped container files with physical
 * memory locking (VirtualLock) and pre-indexed binary layer tables to eliminate
 * OS page-faulting and user-kernel transition overhead during model weight streaming.
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

/**
 * @struct ArcMmapContext
 * @brief Context structure managing the memory-mapped file handles and layer table.
 */
typedef struct {
    void* file_handle;
    void* map_handle;
    uint8_t* base_ptr;
    uint64_t file_size;
    uint32_t total_layers;
    ArcLayerDescriptor layers[ARC_MAX_LAYERS];
    bool is_locked;
} ArcMmapContext;

/**
 * @brief Open and memory-map a model container file using Win32 API.
 *
 * @param file_path Filesystem path to the model container.
 * @return ArcMmapContext* Initialized context pointer, or NULL on failure.
 */
ARC_EXPORT ArcMmapContext* arc_mmap_open(const char* file_path);

/**
 * @brief Register a layer descriptor into the context's binary lookup table.
 *
 * @param ctx Context handle returned by arc_mmap_open.
 * @param layer Pointer to the layer descriptor to register.
 * @return int 0 on success, non-zero error code on failure.
 */
ARC_EXPORT int arc_mmap_register_layer(ArcMmapContext* ctx, const ArcLayerDescriptor* layer);

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
);

/**
 * @brief Lock a memory range into physical RAM to prevent OS paging thrash.
 *
 * @param ctx Context handle returned by arc_mmap_open.
 * @param offset Byte offset from the base mapping pointer.
 * @param size Number of bytes to lock into physical memory.
 * @return int 0 on success, non-zero if VirtualLock fails.
 */
ARC_EXPORT int arc_mmap_lock_range(ArcMmapContext* ctx, uint64_t offset, uint64_t size);

/**
 * @brief Unlock a previously locked physical memory range.
 *
 * @param ctx Context handle returned by arc_mmap_open.
 * @param offset Byte offset from the base mapping pointer.
 * @param size Number of bytes to unlock.
 * @return int 0 on success, non-zero on error.
 */
ARC_EXPORT int arc_mmap_unlock_range(ArcMmapContext* ctx, uint64_t offset, uint64_t size);

/**
 * @brief Unmap views and close all operating system file handles.
 *
 * @param ctx Context handle to close and deallocate.
 */
ARC_EXPORT void arc_mmap_close(ArcMmapContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* ARC_FLASH_STREAM_MMAP_H */
