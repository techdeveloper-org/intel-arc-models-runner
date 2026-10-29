/**
 * @file arc_ze_types.h
 * @brief Level Zero type definitions used by the bridge, copied from ze_api.h (oneapi-src/level-zero).
 *
 * Only the fields and enumerators the bridge needs are declared. Values were checked against
 * the upstream ze_api.h; the struct layouts must match it exactly because the driver writes
 * whole structures through the pointers passed in.
 */

#ifndef ARC_ZE_TYPES_H
#define ARC_ZE_TYPES_H

#include <stddef.h>
#include <stdint.h>

typedef uint32_t ze_result_t;

#define ZE_RESULT_SUCCESS 0u
#define ZE_RESULT_NOT_READY 1u
#define ZE_RESULT_ERROR_DEVICE_LOST 0x70000001u

#define ZE_INIT_FLAG_GPU_ONLY 1u

#define ZE_STYPE_DEVICE_PROPERTIES 0x3u
#define ZE_STYPE_DEVICE_MEMORY_PROPERTIES 0x7u
#define ZE_STYPE_CONTEXT_DESC 0xdu
#define ZE_STYPE_COMMAND_QUEUE_DESC 0xeu
#define ZE_STYPE_COMMAND_LIST_DESC 0xfu
#define ZE_STYPE_DEVICE_MEM_ALLOC_DESC 0x15u
#define ZE_STYPE_HOST_MEM_ALLOC_DESC 0x16u

#define ZE_DEVICE_TYPE_GPU 1u
#define ZE_DEVICE_TYPE_VPU 5u

#define ZE_COMMAND_QUEUE_MODE_DEFAULT 0u
#define ZE_COMMAND_QUEUE_PRIORITY_NORMAL 0u

#define ZE_MAX_DEVICE_NAME 256
#define ZE_MAX_DEVICE_UUID_SIZE 16

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

typedef struct {
    uint32_t stype;
    void* pNext;
    uint32_t type;
    uint32_t vendorId;
    uint32_t deviceId;
    uint32_t flags;
    uint32_t subdeviceId;
    uint32_t coreClockRate;
    uint64_t maxMemAllocSize;
    uint32_t maxHardwareContexts;
    uint32_t maxCommandQueuePriority;
    uint32_t numThreadsPerEU;
    uint32_t physicalEUSimdWidth;
    uint32_t numEUsPerSubslice;
    uint32_t numSubslicesPerSlice;
    uint32_t numSlices;
    uint64_t timerResolution;
    uint32_t timestampValidBits;
    uint32_t kernelTimestampValidBits;
    uint8_t uuid[ZE_MAX_DEVICE_UUID_SIZE];
    char name[ZE_MAX_DEVICE_NAME];
} ze_device_properties_t;

typedef ze_result_t (*pfn_zeInit)(uint32_t flags);
typedef ze_result_t (*pfn_zeDriverGet)(uint32_t* pCount, void** phDrivers);
typedef ze_result_t (*pfn_zeDeviceGet)(void* hDriver, uint32_t* pCount, void** phDevices);
typedef ze_result_t (*pfn_zeDeviceGetProperties)(void* hDevice, ze_device_properties_t* pProperties);
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

#endif /* ARC_ZE_TYPES_H */
