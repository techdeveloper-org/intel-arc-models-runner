/**
 * @file arc_level_zero_bridge.h
 * @brief Bare-metal Level Zero USM allocation and asynchronous DMA streaming interface.
 *
 * Exposes direct access to oneAPI Level Zero Unified Shared Memory (USM) host,
 * device, and shared allocations, alongside asynchronous DMA memory transfer queues
 * targeting Intel Arc Xe-LPG graphics silicon.
 */

#ifndef ARC_LEVEL_ZERO_BRIDGE_H
#define ARC_LEVEL_ZERO_BRIDGE_H

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

/**
 * @enum ArcUsmType
 * @brief Memory residency type for Level Zero Unified Shared Memory buffers.
 */
typedef enum {
    ARC_USM_HOST = 0,
    ARC_USM_DEVICE = 1,
    ARC_USM_SHARED = 2
} ArcUsmType;

/**
 * @struct ArcZeContext
 * @brief Internal runtime state holding Level Zero driver, device, context, and queues.
 */
typedef struct {
    void* ze_module;
    void* driver_handle;
    void* device_handle;
    void* context_handle;
    void* queue_handle;
    void* cmdlist_handle;
    bool is_initialized;
} ArcZeContext;

/**
 * @brief Initialize the Level Zero driver and acquire primary GPU device handles.
 *
 * @return ArcZeContext* Initialized context pointer, or NULL on driver unavailability.
 */
ARC_EXPORT ArcZeContext* arc_ze_init(void);

/**
 * @brief Allocate a Unified Shared Memory (USM) buffer on the designated target.
 *
 * @param ctx Level Zero context initialized by arc_ze_init.
 * @param size_bytes Allocation byte size.
 * @param usm_type Residency type (HOST, DEVICE, or SHARED).
 * @return void* Pointer to the allocated memory buffer, or NULL on failure.
 */
ARC_EXPORT void* arc_ze_alloc_usm(ArcZeContext* ctx, size_t size_bytes, ArcUsmType usm_type);

/**
 * @brief Submit an asynchronous hardware DMA memory copy operation.
 *
 * @param ctx Level Zero context.
 * @param dst Destination memory address.
 * @param src Source memory address.
 * @param size_bytes Number of bytes to transfer via DMA.
 * @return int 0 on successful submission, non-zero on error.
 */
ARC_EXPORT int arc_ze_dma_copy_async(ArcZeContext* ctx, void* dst, const void* src, size_t size_bytes);

/**
 * @brief Synchronize the command queue and block until pending DMA operations finish.
 *
 * @param ctx Level Zero context.
 * @return int 0 on success, non-zero on timeout or execution fault.
 */
ARC_EXPORT int arc_ze_synchronize(ArcZeContext* ctx);

/**
 * @brief Free a previously allocated Unified Shared Memory buffer.
 *
 * @param ctx Level Zero context.
 * @param ptr Pointer to the USM buffer to deallocate.
 */
ARC_EXPORT void arc_ze_free_usm(ArcZeContext* ctx, void* ptr);

/**
 * @brief Release all Level Zero driver resources, queues, and context handles.
 *
 * @param ctx Level Zero context to destroy.
 */
ARC_EXPORT void arc_ze_destroy(ArcZeContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* ARC_LEVEL_ZERO_BRIDGE_H */
