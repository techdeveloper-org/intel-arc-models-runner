/**
 * @file arc_level_zero_bridge.c
 * @brief Dynamic loader implementation for Level Zero USM and asynchronous DMA queues.
 */

#include "arc_level_zero_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
  #include <windows.h>
  #define LOAD_LIB(name) (void*)LoadLibraryA(name)
  #define GET_SYM(h, name) (void*)GetProcAddress((HMODULE)(h), name)
  #define FREE_LIB(h) FreeLibrary((HMODULE)(h))
#else
  #include <dlfcn.h>
  #define LOAD_LIB(name) dlopen(name, RTLD_NOW)
  #define GET_SYM(h, name) dlsym(h, name)
  #define FREE_LIB(h) dlclose(h)
#endif

typedef uint32_t ze_result_t;
#define ZE_RESULT_SUCCESS 0

typedef struct {
    uint32_t stype;
    const void* pNext;
    uint32_t flags;
} ze_context_desc_t;

typedef struct {
    uint32_t stype;
    const void* pNext;
    uint32_t ordinal;
    uint32_t index;
    uint32_t flags;
    uint32_t mode;
    uint32_t priority;
} ze_command_queue_desc_t;

typedef struct {
    uint32_t stype;
    const void* pNext;
    uint32_t commandQueueGroupOrdinal;
    uint32_t flags;
} ze_command_list_desc_t;

typedef struct {
    uint32_t stype;
    const void* pNext;
    uint32_t flags;
} ze_host_mem_alloc_desc_t;

typedef struct {
    uint32_t stype;
    const void* pNext;
    uint32_t flags;
    uint32_t ordinal;
} ze_device_mem_alloc_desc_t;

typedef ze_result_t (*pfn_zeInit)(uint32_t flags);
typedef ze_result_t (*pfn_zeDriverGet)(uint32_t* pCount, void** phDrivers);
typedef ze_result_t (*pfn_zeDeviceGet)(void* hDriver, uint32_t* pCount, void** phDevices);
typedef ze_result_t (*pfn_zeContextCreate)(void* hDriver, const ze_context_desc_t* desc, void** phContext);
typedef ze_result_t (*pfn_zeContextDestroy)(void* hContext);
typedef ze_result_t (*pfn_zeCommandQueueCreate)(void* hContext, void* hDevice, const ze_command_queue_desc_t* desc, void** phCommandQueue);
typedef ze_result_t (*pfn_zeCommandQueueDestroy)(void* hCommandQueue);
typedef ze_result_t (*pfn_zeCommandListCreate)(void* hContext, void* hDevice, const ze_command_list_desc_t* desc, void** phCommandList);
typedef ze_result_t (*pfn_zeCommandListDestroy)(void* hCommandList);
typedef ze_result_t (*pfn_zeCommandListClose)(void* hCommandList);
typedef ze_result_t (*pfn_zeCommandListReset)(void* hCommandList);
typedef ze_result_t (*pfn_zeCommandQueueExecuteCommandLists)(void* hCommandQueue, uint32_t numCommandLists, void** phCommandLists, void* hFence);
typedef ze_result_t (*pfn_zeCommandQueueSynchronize)(void* hCommandQueue, uint64_t timeout);
typedef ze_result_t (*pfn_zeMemAllocHost)(void* hContext, const ze_host_mem_alloc_desc_t* host_desc, size_t size, size_t alignment, void** pptr);
typedef ze_result_t (*pfn_zeMemAllocDevice)(void* hContext, const ze_device_mem_alloc_desc_t* device_desc, size_t size, size_t alignment, void* hDevice, void** pptr);
typedef ze_result_t (*pfn_zeMemAllocShared)(void* hContext, const ze_device_mem_alloc_desc_t* device_desc, const ze_host_mem_alloc_desc_t* host_desc, size_t size, size_t alignment, void* hDevice, void** pptr);
typedef ze_result_t (*pfn_zeMemFree)(void* hContext, void* ptr);
typedef ze_result_t (*pfn_zeCommandListAppendMemoryCopy)(void* hCommandList, void* dstptr, const void* srcptr, size_t size, void* hEvent, uint32_t numWaitEvents, void** phWaitEvents);

static pfn_zeInit fn_zeInit = NULL;
static pfn_zeDriverGet fn_zeDriverGet = NULL;
static pfn_zeDeviceGet fn_zeDeviceGet = NULL;
static pfn_zeContextCreate fn_zeContextCreate = NULL;
static pfn_zeContextDestroy fn_zeContextDestroy = NULL;
static pfn_zeCommandQueueCreate fn_zeCommandQueueCreate = NULL;
static pfn_zeCommandQueueDestroy fn_zeCommandQueueDestroy = NULL;
static pfn_zeCommandListCreate fn_zeCommandListCreate = NULL;
static pfn_zeCommandListDestroy fn_zeCommandListDestroy = NULL;
static pfn_zeCommandListClose fn_zeCommandListClose = NULL;
static pfn_zeCommandListReset fn_zeCommandListReset = NULL;
static pfn_zeCommandQueueExecuteCommandLists fn_zeCommandQueueExecuteCommandLists = NULL;
static pfn_zeCommandQueueSynchronize fn_zeCommandQueueSynchronize = NULL;
static pfn_zeMemAllocHost fn_zeMemAllocHost = NULL;
static pfn_zeMemAllocDevice fn_zeMemAllocDevice = NULL;
static pfn_zeMemAllocShared fn_zeMemAllocShared = NULL;
static pfn_zeMemFree fn_zeMemFree = NULL;
static pfn_zeCommandListAppendMemoryCopy fn_zeCommandListAppendMemoryCopy = NULL;

/**
 * @brief Initialize the Level Zero driver and acquire primary GPU device handles.
 *
 * @return ArcZeContext* Initialized context pointer, or NULL on driver unavailability.
 */
ARC_EXPORT ArcZeContext* arc_ze_init(void) {
    ArcZeContext* ctx = (ArcZeContext*)calloc(1, sizeof(ArcZeContext));
    if (!ctx) {
        return NULL;
    }

#if defined(_WIN32)
    const char* candidate_libs[] = {"ze_loader.dll", NULL};
#else
    const char* candidate_libs[] = {"libze_loader.so.1", "libze_loader.so", NULL};
#endif

    void* hMod = NULL;
    for (int i = 0; candidate_libs[i] != NULL; ++i) {
        hMod = LOAD_LIB(candidate_libs[i]);
        if (hMod) {
            break;
        }
    }

    if (!hMod) {
        free(ctx);
        return NULL;
    }

    ctx->ze_module = hMod;

    fn_zeInit = (pfn_zeInit)GET_SYM(hMod, "zeInit");
    fn_zeDriverGet = (pfn_zeDriverGet)GET_SYM(hMod, "zeDriverGet");
    fn_zeDeviceGet = (pfn_zeDeviceGet)GET_SYM(hMod, "zeDeviceGet");
    fn_zeContextCreate = (pfn_zeContextCreate)GET_SYM(hMod, "zeContextCreate");
    fn_zeContextDestroy = (pfn_zeContextDestroy)GET_SYM(hMod, "zeContextDestroy");
    fn_zeCommandQueueCreate = (pfn_zeCommandQueueCreate)GET_SYM(hMod, "zeCommandQueueCreate");
    fn_zeCommandQueueDestroy = (pfn_zeCommandQueueDestroy)GET_SYM(hMod, "zeCommandQueueDestroy");
    fn_zeCommandListCreate = (pfn_zeCommandListCreate)GET_SYM(hMod, "zeCommandListCreate");
    fn_zeCommandListDestroy = (pfn_zeCommandListDestroy)GET_SYM(hMod, "zeCommandListDestroy");
    fn_zeCommandListClose = (pfn_zeCommandListClose)GET_SYM(hMod, "zeCommandListClose");
    fn_zeCommandListReset = (pfn_zeCommandListReset)GET_SYM(hMod, "zeCommandListReset");
    fn_zeCommandQueueExecuteCommandLists = (pfn_zeCommandQueueExecuteCommandLists)GET_SYM(hMod, "zeCommandQueueExecuteCommandLists");
    fn_zeCommandQueueSynchronize = (pfn_zeCommandQueueSynchronize)GET_SYM(hMod, "zeCommandQueueSynchronize");
    fn_zeMemAllocHost = (pfn_zeMemAllocHost)GET_SYM(hMod, "zeMemAllocHost");
    fn_zeMemAllocDevice = (pfn_zeMemAllocDevice)GET_SYM(hMod, "zeMemAllocDevice");
    fn_zeMemAllocShared = (pfn_zeMemAllocShared)GET_SYM(hMod, "zeMemAllocShared");
    fn_zeMemFree = (pfn_zeMemFree)GET_SYM(hMod, "zeMemFree");
    fn_zeCommandListAppendMemoryCopy = (pfn_zeCommandListAppendMemoryCopy)GET_SYM(hMod, "zeCommandListAppendMemoryCopy");

    if (!fn_zeInit || !fn_zeDriverGet || !fn_zeDeviceGet || !fn_zeContextCreate || !fn_zeMemAllocShared) {
        FREE_LIB(hMod);
        free(ctx);
        return NULL;
    }

    if (fn_zeInit(0) != ZE_RESULT_SUCCESS) {
        FREE_LIB(hMod);
        free(ctx);
        return NULL;
    }

    uint32_t driverCount = 0;
    if (fn_zeDriverGet(&driverCount, NULL) != ZE_RESULT_SUCCESS || driverCount == 0) {
        FREE_LIB(hMod);
        free(ctx);
        return NULL;
    }

    void** drivers = (void**)malloc(sizeof(void*) * driverCount);
    fn_zeDriverGet(&driverCount, drivers);
    ctx->driver_handle = drivers[0];
    free(drivers);

    uint32_t deviceCount = 0;
    if (fn_zeDeviceGet(ctx->driver_handle, &deviceCount, NULL) != ZE_RESULT_SUCCESS || deviceCount == 0) {
        FREE_LIB(hMod);
        free(ctx);
        return NULL;
    }

    void** devices = (void**)malloc(sizeof(void*) * deviceCount);
    fn_zeDeviceGet(ctx->driver_handle, &deviceCount, devices);
    ctx->device_handle = devices[0];
    free(devices);

    ze_context_desc_t contextDesc = {0};
    contextDesc.stype = 0x1;
    if (fn_zeContextCreate(ctx->driver_handle, &contextDesc, &ctx->context_handle) != ZE_RESULT_SUCCESS) {
        FREE_LIB(hMod);
        free(ctx);
        return NULL;
    }

    ze_command_queue_desc_t queueDesc = {0};
    queueDesc.stype = 0x2;
    queueDesc.mode = 0;
    queueDesc.priority = 1;
    fn_zeCommandQueueCreate(ctx->context_handle, ctx->device_handle, &queueDesc, &ctx->queue_handle);

    ze_command_list_desc_t listDesc = {0};
    listDesc.stype = 0x3;
    fn_zeCommandListCreate(ctx->context_handle, ctx->device_handle, &listDesc, &ctx->cmdlist_handle);

    ctx->is_initialized = true;
    return ctx;
}

/**
 * @brief Allocate a Unified Shared Memory (USM) buffer on the designated target.
 *
 * @param ctx Level Zero context initialized by arc_ze_init.
 * @param size_bytes Allocation byte size.
 * @param usm_type Residency type (HOST, DEVICE, or SHARED).
 * @return void* Pointer to the allocated memory buffer, or NULL on failure.
 */
ARC_EXPORT void* arc_ze_alloc_usm(ArcZeContext* ctx, size_t size_bytes, ArcUsmType usm_type) {
    if (!ctx || !ctx->is_initialized || !ctx->context_handle || size_bytes == 0) {
        return NULL;
    }

    void* ptr = NULL;
    ze_result_t res = 1;

    if (usm_type == ARC_USM_HOST && fn_zeMemAllocHost) {
        ze_host_mem_alloc_desc_t desc = {0};
        desc.stype = 0x4;
        res = fn_zeMemAllocHost(ctx->context_handle, &desc, size_bytes, 64, &ptr);
    } else if (usm_type == ARC_USM_DEVICE && fn_zeMemAllocDevice) {
        ze_device_mem_alloc_desc_t desc = {0};
        desc.stype = 0x5;
        res = fn_zeMemAllocDevice(ctx->context_handle, &desc, size_bytes, 64, ctx->device_handle, &ptr);
    } else if (usm_type == ARC_USM_SHARED && fn_zeMemAllocShared) {
        ze_device_mem_alloc_desc_t devDesc = {0};
        devDesc.stype = 0x5;
        ze_host_mem_alloc_desc_t hostDesc = {0};
        hostDesc.stype = 0x4;
        res = fn_zeMemAllocShared(ctx->context_handle, &devDesc, &hostDesc, size_bytes, 64, ctx->device_handle, &ptr);
    }

    if (res != ZE_RESULT_SUCCESS) {
        return NULL;
    }

    return ptr;
}

/**
 * @brief Submit an asynchronous hardware DMA memory copy operation.
 *
 * @param ctx Level Zero context.
 * @param dst Destination memory address.
 * @param src Source memory address.
 * @param size_bytes Number of bytes to transfer via DMA.
 * @return int 0 on successful submission, non-zero on error.
 */
ARC_EXPORT int arc_ze_dma_copy_async(ArcZeContext* ctx, void* dst, const void* src, size_t size_bytes) {
    if (!ctx || !ctx->cmdlist_handle || !dst || !src || size_bytes == 0) {
        return -1;
    }

    if (!fn_zeCommandListAppendMemoryCopy) {
        memcpy(dst, src, size_bytes);
        return 0;
    }

    ze_result_t res = fn_zeCommandListAppendMemoryCopy(
        ctx->cmdlist_handle,
        dst,
        src,
        size_bytes,
        NULL,
        0,
        NULL
    );

    return (res == ZE_RESULT_SUCCESS) ? 0 : (int)res;
}

/**
 * @brief Synchronize the command queue and block until pending DMA operations finish.
 *
 * @param ctx Level Zero context.
 * @return int 0 on success, non-zero on timeout or execution fault.
 */
ARC_EXPORT int arc_ze_synchronize(ArcZeContext* ctx) {
    if (!ctx || !ctx->queue_handle) {
        return -1;
    }

    if (ctx->cmdlist_handle && fn_zeCommandListClose && fn_zeCommandQueueExecuteCommandLists) {
        fn_zeCommandListClose(ctx->cmdlist_handle);
        fn_zeCommandQueueExecuteCommandLists(ctx->queue_handle, 1, &ctx->cmdlist_handle, NULL);
    }

    if (fn_zeCommandQueueSynchronize) {
        ze_result_t res = fn_zeCommandQueueSynchronize(ctx->queue_handle, 10000000000ULL);
        if (ctx->cmdlist_handle && fn_zeCommandListReset) {
            fn_zeCommandListReset(ctx->cmdlist_handle);
        }
        return (res == ZE_RESULT_SUCCESS) ? 0 : (int)res;
    }

    return 0;
}

/**
 * @brief Free a previously allocated Unified Shared Memory buffer.
 *
 * @param ctx Level Zero context.
 * @param ptr Pointer to the USM buffer to deallocate.
 */
ARC_EXPORT void arc_ze_free_usm(ArcZeContext* ctx, void* ptr) {
    if (!ctx || !ctx->context_handle || !ptr) {
        return;
    }

    if (fn_zeMemFree) {
        fn_zeMemFree(ctx->context_handle, ptr);
    }
}

/**
 * @brief Release all Level Zero driver resources, queues, and context handles.
 *
 * @param ctx Level Zero context to destroy.
 */
ARC_EXPORT void arc_ze_destroy(ArcZeContext* ctx) {
    if (!ctx) {
        return;
    }

    if (ctx->cmdlist_handle && fn_zeCommandListDestroy) {
        fn_zeCommandListDestroy(ctx->cmdlist_handle);
        ctx->cmdlist_handle = NULL;
    }

    if (ctx->queue_handle && fn_zeCommandQueueDestroy) {
        fn_zeCommandQueueDestroy(ctx->queue_handle);
        ctx->queue_handle = NULL;
    }

    if (ctx->context_handle && fn_zeContextDestroy) {
        fn_zeContextDestroy(ctx->context_handle);
        ctx->context_handle = NULL;
    }

    if (ctx->ze_module) {
        FREE_LIB(ctx->ze_module);
        ctx->ze_module = NULL;
    }

    free(ctx);
}
