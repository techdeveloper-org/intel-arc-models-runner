/**
 * @file fake_ze_loader.c
 * @brief Minimal fake Level Zero loader used to test arc_level_zero_bridge.c without hardware.
 *
 * Behaviour is controlled by environment variables read at call time:
 *   FAKE_ZE_LAYOUT   "5;1" device types per driver (default NPU driver first, GPU second)
 *   FAKE_ZE_FAIL     name of one function that must fail
 *   FAKE_ZE_SYNC     ok | never | lost | slow:<ms>
 *   FAKE_ZE_COPY_FAIL=1  make zeCommandListAppendMemoryCopy fail
 * Every descriptor structure type (stype) is checked against the values in ze_api.h; a wrong
 * value makes the call fail with an invalid-enumeration result.
 */

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "../../src/arc_runner/native/arc_ze_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Expected structure types, copied from upstream ze_api.h (oneapi-src/level-zero) independently of
 * the bridge's own header so that a wrong constant in the bridge is detected. */
#define FAKE_STYPE_DEVICE_PROPERTIES 0x3u
#define FAKE_STYPE_DEVICE_MEMORY_PROPERTIES 0x7u
#define FAKE_STYPE_CONTEXT_DESC 0xdu
#define FAKE_STYPE_COMMAND_QUEUE_DESC 0xeu
#define FAKE_STYPE_COMMAND_LIST_DESC 0xfu
#define FAKE_STYPE_DEVICE_MEM_ALLOC_DESC 0x15u
#define FAKE_STYPE_HOST_MEM_ALLOC_DESC 0x16u
#define FAKE_ERR_INVALID_ENUM 0x78000004u
#define FAKE_ERR_FAILED 0x70000004u
#define FAKE_ERR_OOM 0x70000002u
#define FAKE_MAX_DRIVERS 4
#define FAKE_MAX_DEVICES 4
#define FAKE_EXPORT __attribute__((visibility("default")))

static unsigned long g_live_allocs = 0;
static unsigned long long g_sync_started_ms = 0;
static int g_sync_started = 0;
static char g_driver_tokens[FAKE_MAX_DRIVERS];
static char g_device_tokens[FAKE_MAX_DEVICES];

static unsigned long long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000u + (unsigned long long)ts.tv_nsec / 1000000u;
}

static int should_fail(const char* name) {
    const char* target = getenv("FAKE_ZE_FAIL");
    return target && strcmp(target, name) == 0;
}

static unsigned layout_driver_count(void) {
    const char* layout = getenv("FAKE_ZE_LAYOUT");
    unsigned count = 1;
    for (const char* c = layout ? layout : "5;1"; *c; ++c) {
        count += (*c == ';');
    }
    return count > FAKE_MAX_DRIVERS ? FAKE_MAX_DRIVERS : count;
}

static unsigned device_type_of_driver(unsigned driver_index) {
    const char* layout = getenv("FAKE_ZE_LAYOUT");
    layout = layout ? layout : "5;1";
    unsigned current = 0;
    while (current < driver_index) {
        const char* semi = strchr(layout, ';');
        if (!semi) {
            return 0;
        }
        layout = semi + 1;
        ++current;
    }
    return (unsigned)strtoul(layout, NULL, 10);
}

FAKE_EXPORT unsigned long fake_ze_live_allocs(void) {
    return g_live_allocs;
}

FAKE_EXPORT ze_result_t zeInit(uint32_t flags) {
    (void)flags;
    return should_fail("zeInit") ? FAKE_ERR_FAILED : ZE_RESULT_SUCCESS;
}

FAKE_EXPORT ze_result_t zeDriverGet(uint32_t* count, void** drivers) {
    if (should_fail("zeDriverGet")) {
        return FAKE_ERR_FAILED;
    }
    unsigned n = layout_driver_count();
    if (!drivers) {
        *count = n;
        return ZE_RESULT_SUCCESS;
    }
    for (unsigned i = 0; i < n && i < *count; ++i) {
        drivers[i] = &g_driver_tokens[i];
    }
    return ZE_RESULT_SUCCESS;
}

FAKE_EXPORT ze_result_t zeDeviceGet(void* driver, uint32_t* count, void** devices) {
    if (should_fail("zeDeviceGet")) {
        return FAKE_ERR_FAILED;
    }
    if (!devices) {
        *count = 1;
        return ZE_RESULT_SUCCESS;
    }
    unsigned index = (unsigned)((char*)driver - g_driver_tokens);
    devices[0] = &g_device_tokens[index];
    return ZE_RESULT_SUCCESS;
}

FAKE_EXPORT ze_result_t zeDeviceGetProperties(void* device, ze_device_properties_t* props) {
    if (should_fail("zeDeviceGetProperties")) {
        return FAKE_ERR_FAILED;
    }
    if (props->stype != FAKE_STYPE_DEVICE_PROPERTIES) {
        return FAKE_ERR_INVALID_ENUM;
    }
    props->type = device_type_of_driver((unsigned)((char*)device - g_device_tokens));
    snprintf(props->name, sizeof props->name, "fake device");
    return ZE_RESULT_SUCCESS;
}

FAKE_EXPORT ze_result_t zeContextCreate(void* driver, const ze_context_desc_t* desc, void** out) {
    (void)driver;
    if (should_fail("zeContextCreate")) {
        return FAKE_ERR_FAILED;
    }
    if (desc->stype != FAKE_STYPE_CONTEXT_DESC) {
        return FAKE_ERR_INVALID_ENUM;
    }
    *out = malloc(8);
    return ZE_RESULT_SUCCESS;
}

FAKE_EXPORT ze_result_t zeContextDestroy(void* context) {
    free(context);
    return ZE_RESULT_SUCCESS;
}

FAKE_EXPORT ze_result_t zeCommandQueueCreate(void* context, void* device, const ze_command_queue_desc_t* desc, void** out) {
    (void)context;
    (void)device;
    if (should_fail("zeCommandQueueCreate")) {
        return FAKE_ERR_FAILED;
    }
    if (desc->stype != FAKE_STYPE_COMMAND_QUEUE_DESC || desc->priority != ZE_COMMAND_QUEUE_PRIORITY_NORMAL) {
        return FAKE_ERR_INVALID_ENUM;
    }
    *out = malloc(8);
    return ZE_RESULT_SUCCESS;
}

FAKE_EXPORT ze_result_t zeCommandQueueDestroy(void* queue) {
    free(queue);
    return ZE_RESULT_SUCCESS;
}

FAKE_EXPORT ze_result_t zeCommandListCreate(void* context, void* device, const ze_command_list_desc_t* desc, void** out) {
    (void)context;
    (void)device;
    if (should_fail("zeCommandListCreate")) {
        return FAKE_ERR_FAILED;
    }
    if (desc->stype != FAKE_STYPE_COMMAND_LIST_DESC) {
        return FAKE_ERR_INVALID_ENUM;
    }
    *out = malloc(8);
    return ZE_RESULT_SUCCESS;
}

FAKE_EXPORT ze_result_t zeCommandListDestroy(void* list) {
    free(list);
    return ZE_RESULT_SUCCESS;
}

FAKE_EXPORT ze_result_t zeCommandListClose(void* list) {
    (void)list;
    return ZE_RESULT_SUCCESS;
}

FAKE_EXPORT ze_result_t zeCommandListReset(void* list) {
    (void)list;
    g_sync_started = 0;
    return ZE_RESULT_SUCCESS;
}

FAKE_EXPORT ze_result_t zeCommandQueueExecuteCommandLists(void* queue, uint32_t n, void** lists, void* fence) {
    (void)queue;
    (void)n;
    (void)lists;
    (void)fence;
    g_sync_started = 0;
    return should_fail("zeCommandQueueExecuteCommandLists") ? FAKE_ERR_FAILED : ZE_RESULT_SUCCESS;
}

FAKE_EXPORT ze_result_t zeCommandQueueSynchronize(void* queue, uint64_t timeout_ns) {
    (void)queue;
    const char* mode = getenv("FAKE_ZE_SYNC");
    mode = mode ? mode : "ok";
    if (strcmp(mode, "ok") == 0) {
        return ZE_RESULT_SUCCESS;
    }
    if (strcmp(mode, "lost") == 0) {
        return ZE_RESULT_ERROR_DEVICE_LOST;
    }
    if (!g_sync_started) {
        g_sync_started = 1;
        g_sync_started_ms = now_ms();
    }
    unsigned long long slice_ms = timeout_ns / 1000000u;
    struct timespec nap = {0, 5 * 1000000L};
    unsigned long long waited = 0;
    unsigned long long finish_after = strncmp(mode, "slow:", 5) == 0 ? strtoull(mode + 5, NULL, 10) : ~0ull;
    while (waited < slice_ms) {
        if (now_ms() - g_sync_started_ms >= finish_after) {
            return ZE_RESULT_SUCCESS;
        }
        nanosleep(&nap, NULL);
        waited += 5;
    }
    return ZE_RESULT_NOT_READY;
}

static ze_result_t fake_alloc(const char* name, size_t size, void** out) {
    if (should_fail(name)) {
        return FAKE_ERR_FAILED;
    }
    if (size > ((size_t)8 << 30)) {
        return FAKE_ERR_OOM;
    }
    *out = malloc(size ? size : 1);
    if (!*out) {
        return FAKE_ERR_OOM;
    }
    ++g_live_allocs;
    return ZE_RESULT_SUCCESS;
}

FAKE_EXPORT ze_result_t zeMemAllocHost(void* context, const ze_host_mem_alloc_desc_t* desc, size_t size, size_t alignment, void** out) {
    (void)context;
    (void)alignment;
    if (desc->stype != FAKE_STYPE_HOST_MEM_ALLOC_DESC) {
        return FAKE_ERR_INVALID_ENUM;
    }
    return fake_alloc("zeMemAllocHost", size, out);
}

FAKE_EXPORT ze_result_t zeMemAllocDevice(void* context, const ze_device_mem_alloc_desc_t* desc, size_t size, size_t alignment, void* device, void** out) {
    (void)context;
    (void)alignment;
    (void)device;
    if (desc->stype != FAKE_STYPE_DEVICE_MEM_ALLOC_DESC) {
        return FAKE_ERR_INVALID_ENUM;
    }
    return fake_alloc("zeMemAllocDevice", size, out);
}

FAKE_EXPORT ze_result_t zeMemAllocShared(void* context, const ze_device_mem_alloc_desc_t* device_desc, const ze_host_mem_alloc_desc_t* host_desc, size_t size, size_t alignment, void* device, void** out) {
    (void)context;
    (void)alignment;
    (void)device;
    if (device_desc->stype != FAKE_STYPE_DEVICE_MEM_ALLOC_DESC || host_desc->stype != FAKE_STYPE_HOST_MEM_ALLOC_DESC) {
        return FAKE_ERR_INVALID_ENUM;
    }
    return fake_alloc("zeMemAllocShared", size, out);
}

FAKE_EXPORT ze_result_t zeMemFree(void* context, void* ptr) {
    (void)context;
    if (ptr) {
        free(ptr);
        --g_live_allocs;
    }
    return ZE_RESULT_SUCCESS;
}

FAKE_EXPORT ze_result_t zeCommandListAppendMemoryCopy(void* list, void* dst, const void* src, size_t size, void* event, uint32_t n_wait, void** waits) {
    (void)list;
    (void)dst;
    (void)src;
    (void)size;
    (void)event;
    (void)n_wait;
    (void)waits;
    const char* flag = getenv("FAKE_ZE_COPY_FAIL");
    return flag && flag[0] == '1' ? FAKE_ERR_FAILED : ZE_RESULT_SUCCESS;
}

typedef struct {
    uint32_t stype;
    void* pNext;
    uint32_t flags;
    uint32_t maxClockRate;
    uint32_t maxBusWidth;
    uint64_t totalSize;
    char name[ZE_MAX_DEVICE_NAME];
} fake_device_memory_properties_t;

FAKE_EXPORT ze_result_t zeDeviceGetMemoryProperties(void* device, uint32_t* count, fake_device_memory_properties_t* props) {
    (void)device;
    if (should_fail("zeDeviceGetMemoryProperties")) {
        return FAKE_ERR_FAILED;
    }
    if (!props) {
        *count = 1;
        return ZE_RESULT_SUCCESS;
    }
    if (props[0].stype != FAKE_STYPE_DEVICE_MEMORY_PROPERTIES) {
        return FAKE_ERR_INVALID_ENUM;
    }
    props[0].totalSize = (uint64_t)8 << 30;
    return ZE_RESULT_SUCCESS;
}
