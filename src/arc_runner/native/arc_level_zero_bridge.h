/**
 * @file arc_level_zero_bridge.h
 * @brief Level Zero USM and DMA bridge loaded dynamically from the system Level Zero loader.
 *
 * The bridge never links against Level Zero. arc_ze_init selects the first device whose type is
 * ZE_DEVICE_TYPE_GPU (the NPU, type VPU, is never selected), validates every driver call, and
 * returns an opaque context. All waits are bounded below the Windows TDR limit (2 s); a wait that
 * times out or reports device loss marks the context dead and every later call fails fast.
 *
 * The USM and DMA functions are used by the optional Python DeviceLayerStager (opt-in on the
 * flash-stream weight manager); the default streaming path does not call them.
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

#define ARC_MAX_TRACKED_ALLOCS 1024
#define ARC_MAX_USM_BYTES ((uint64_t)1 << 36)
#define ARC_ZE_DEFAULT_SYNC_TIMEOUT_MS 1500u
#define ARC_ZE_MAX_SYNC_TIMEOUT_MS 1900u
#define ARC_ZE_LOADER_ENV "ARC_ZE_LOADER_PATH"
#define ARC_ZE_SYNC_TIMEOUT_ENV "ARC_ZE_SYNC_TIMEOUT_MS"

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
 * @enum ArcZeStatus
 * @brief Result codes of the arc_ze_* functions; negative values are errors.
 */
typedef enum ArcZeStatus {
    ARC_ZE_OK = 0,
    ARC_ZE_FALLBACK_MEMCPY = 1,
    ARC_ZE_ERR_INVALID_ARG = -1,
    ARC_ZE_ERR_LOAD = -2,
    ARC_ZE_ERR_SYMBOL = -3,
    ARC_ZE_ERR_NO_GPU = -4,
    ARC_ZE_ERR_API = -5,
    ARC_ZE_ERR_TOO_LARGE = -6,
    ARC_ZE_ERR_TOO_MANY_ALLOCS = -7,
    ARC_ZE_ERR_DEAD = -8,
    ARC_ZE_ERR_TIMEOUT = -9,
    ARC_ZE_ERR_NOT_INITIALIZED = -10
} ArcZeStatus;

/**
 * @struct ArcZeContext
 * @brief Opaque runtime state holding the Level Zero module, device, context, queue and list.
 */
typedef struct ArcZeContext ArcZeContext;

/**
 * @brief Initialize Level Zero and acquire the first GPU device (never the NPU).
 *
 * The loader library is taken from the ARC_ZE_LOADER_PATH environment variable if set,
 * otherwise from the platform default names.
 *
 * @param out_status Optional receiver for an ArcZeStatus describing the failure.
 * @return Context pointer, or NULL if no usable GPU driver was found or any driver call failed.
 */
ARC_EXPORT ArcZeContext* arc_ze_init_ex(int32_t* out_status);

/** @brief Same as arc_ze_init_ex without a status receiver. */
ARC_EXPORT ArcZeContext* arc_ze_init(void);

/**
 * @brief Allocate a USM buffer.
 *
 * @param ctx Context from arc_ze_init.
 * @param size_bytes Byte size; must satisfy 0 < size_bytes <= ARC_MAX_USM_BYTES.
 * @param usm_type Residency type.
 * @return Pointer, or NULL with the reason in arc_ze_last_status.
 */
ARC_EXPORT void* arc_ze_alloc_usm(ArcZeContext* ctx, size_t size_bytes, ArcUsmType usm_type);

/**
 * @brief Record a copy on the command list.
 *
 * If the driver rejects the copy and both pointers are tracked host or shared allocations, the
 * copy is done with memcpy and ARC_ZE_FALLBACK_MEMCPY is returned.
 *
 * @param ctx Context from arc_ze_init.
 * @param dst Destination address.
 * @param src Source address.
 * @param size_bytes Byte count.
 * @return ARC_ZE_OK, ARC_ZE_FALLBACK_MEMCPY, or a negative ArcZeStatus.
 */
ARC_EXPORT int arc_ze_dma_copy_async(ArcZeContext* ctx, void* dst, const void* src, size_t size_bytes);

/**
 * @brief Execute the recorded commands and wait for them, polling in slices of at most 250 ms.
 *
 * On timeout or device loss the context is marked dead, the command list is not reset, and
 * ARC_ZE_ERR_TIMEOUT or ARC_ZE_ERR_API is returned.
 *
 * @param ctx Context from arc_ze_init.
 * @return ARC_ZE_OK or a negative ArcZeStatus.
 */
ARC_EXPORT int arc_ze_synchronize(ArcZeContext* ctx);

/**
 * @brief Free a tracked USM allocation; unknown pointers and NULL are ignored.
 *
 * @param ctx Context from arc_ze_init.
 * @param ptr Pointer returned by arc_ze_alloc_usm.
 */
ARC_EXPORT void arc_ze_free_usm(ArcZeContext* ctx, void* ptr);

/**
 * @brief Free outstanding allocations and release all Level Zero resources; NULL is allowed.
 *
 * @param ctx Context to destroy.
 */
ARC_EXPORT void arc_ze_destroy(ArcZeContext* ctx);

/** @brief Return 1 if the context was marked dead (timeout or device loss), else 0. */
ARC_EXPORT int arc_ze_is_dead(const ArcZeContext* ctx);

/** @brief Return the last ArcZeStatus recorded on the context. */
ARC_EXPORT int32_t arc_ze_last_status(const ArcZeContext* ctx);

/** @brief Return the last raw ze_result_t returned by the driver. */
ARC_EXPORT uint32_t arc_ze_last_result(const ArcZeContext* ctx);

/** @brief Return the ze_device_type_t of the selected device (1 = GPU), or 0 for NULL. */
ARC_EXPORT uint32_t arc_ze_device_type(const ArcZeContext* ctx);

/** @brief Return the number of live tracked allocations. */
ARC_EXPORT uint32_t arc_ze_alloc_count(const ArcZeContext* ctx);

/** @brief Return the configured synchronize timeout in milliseconds. */
ARC_EXPORT uint32_t arc_ze_sync_timeout_ms(const ArcZeContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* ARC_LEVEL_ZERO_BRIDGE_H */
