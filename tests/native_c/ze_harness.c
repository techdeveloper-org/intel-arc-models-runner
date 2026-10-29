/**
 * @file ze_harness.c
 * @brief Sanitizer-backed tests for arc_level_zero_bridge.c against a fake Level Zero loader.
 *
 * Usage: ze_harness <path-to-fake-loader.so>
 */

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "../../src/arc_runner/native/arc_level_zero_bridge.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static unsigned g_failures = 0;
static unsigned g_checks = 0;
static unsigned long (*g_live_allocs)(void) = NULL;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failures;                                                        \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
        }                                                                        \
    } while (0)

static unsigned long long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000u + (unsigned long long)ts.tv_nsec / 1000000u;
}

/**
 * @brief Reset all fake-loader controls to their defaults.
 */
static void reset_env(void) {
    unsetenv("FAKE_ZE_LAYOUT");
    unsetenv("FAKE_ZE_FAIL");
    unsetenv("FAKE_ZE_SYNC");
    unsetenv("FAKE_ZE_COPY_FAIL");
    unsetenv("ARC_ZE_SYNC_TIMEOUT_MS");
}

/**
 * @brief Initialize a context and require success.
 *
 * @return Context pointer (never NULL when the check passes).
 */
static ArcZeContext* init_ok(void) {
    int32_t status = -99;
    ArcZeContext* ctx = arc_ze_init_ex(&status);
    CHECK(ctx != NULL && status == ARC_ZE_OK);
    return ctx;
}

static void test_selects_gpu_not_npu(void) {
    reset_env();
    ArcZeContext* ctx = init_ok();
    CHECK(ctx && arc_ze_device_type(ctx) == 1u);
    CHECK(ctx && arc_ze_is_dead(ctx) == 0);
    CHECK(ctx && arc_ze_sync_timeout_ms(ctx) == ARC_ZE_DEFAULT_SYNC_TIMEOUT_MS);
    arc_ze_destroy(ctx);

    setenv("FAKE_ZE_LAYOUT", "1;5", 1);
    ctx = init_ok();
    CHECK(ctx && arc_ze_device_type(ctx) == 1u);
    arc_ze_destroy(ctx);

    int32_t status = 0;
    setenv("FAKE_ZE_LAYOUT", "5", 1);
    CHECK(arc_ze_init_ex(&status) == NULL && status == ARC_ZE_ERR_NO_GPU);
    setenv("FAKE_ZE_LAYOUT", "5;2;3", 1);
    CHECK(arc_ze_init_ex(&status) == NULL && status == ARC_ZE_ERR_NO_GPU);
    CHECK(arc_ze_init() == NULL);
}

static void test_init_reports_every_failing_driver_call(void) {
    static const char* const names[] = {
        "zeInit", "zeDriverGet", "zeDeviceGet", "zeDeviceGetProperties", "zeContextCreate",
        "zeCommandQueueCreate", "zeCommandListCreate",
    };
    for (size_t i = 0; i < sizeof names / sizeof names[0]; ++i) {
        reset_env();
        setenv("FAKE_ZE_FAIL", names[i], 1);
        int32_t status = 0;
        ArcZeContext* ctx = arc_ze_init_ex(&status);
        int ok = ctx == NULL && (status == ARC_ZE_ERR_API || status == ARC_ZE_ERR_NO_GPU);
        if (!ok) {
            fprintf(stderr, "init did not fail cleanly when %s fails (status=%d)\n", names[i], status);
        }
        CHECK(ok);
        if (ctx) {
            arc_ze_destroy(ctx);
        }
    }
    reset_env();
}

static void test_missing_loader_and_symbols(void) {
    reset_env();
    const char* saved = getenv("ARC_ZE_LOADER_PATH");
    char* keep = saved ? strdup(saved) : NULL;
    int32_t status = 0;
    setenv("ARC_ZE_LOADER_PATH", "/nonexistent/libze_loader.so", 1);
    CHECK(arc_ze_init_ex(&status) == NULL && status == ARC_ZE_ERR_LOAD);
    setenv("ARC_ZE_LOADER_PATH", "libc.so.6", 1);
    CHECK(arc_ze_init_ex(&status) == NULL && status == ARC_ZE_ERR_SYMBOL);
    if (keep) {
        setenv("ARC_ZE_LOADER_PATH", keep, 1);
        free(keep);
    }
}

/**
 * @brief Verify the ARC_ZE_LOADER_PATH policy (TM-06): only a regular file named like a Level Zero
 * loader is accepted, unless ARC_ZE_LOADER_ALLOW_ANY is exactly "1".
 *
 * @param fake_path Path of the fake loader library, which is not named like a Level Zero loader.
 */
static void test_loader_override_policy(const char* fake_path) {
    reset_env();
    char dir_template[] = "/tmp/ze_policy_XXXXXX";
    char* dir = mkdtemp(dir_template);
    CHECK(dir != NULL);
    if (!dir) {
        return;
    }
    char good[512], bad_name[512], dir_named[512];
    snprintf(good, sizeof good, "%s/libze_loader.so", dir);
    snprintf(bad_name, sizeof bad_name, "%s/libevil_loader.so", dir);
    snprintf(dir_named, sizeof dir_named, "%s/ze_loader.dll", dir);
    CHECK(symlink(fake_path, good) == 0);
    CHECK(symlink(fake_path, bad_name) == 0);
    CHECK(mkdir(dir_named, 0700) == 0);

    unsetenv("ARC_ZE_LOADER_ALLOW_ANY");
    int32_t status = 0;
    setenv("ARC_ZE_LOADER_PATH", fake_path, 1);
    CHECK(arc_ze_init_ex(&status) == NULL && status == ARC_ZE_ERR_LOAD);
    setenv("ARC_ZE_LOADER_PATH", bad_name, 1);
    CHECK(arc_ze_init_ex(&status) == NULL && status == ARC_ZE_ERR_LOAD);
    setenv("ARC_ZE_LOADER_PATH", dir_named, 1);
    CHECK(arc_ze_init_ex(&status) == NULL && status == ARC_ZE_ERR_LOAD);
    setenv("ARC_ZE_LOADER_PATH", "libc.so.6", 1);
    CHECK(arc_ze_init_ex(&status) == NULL && status == ARC_ZE_ERR_LOAD);

    setenv("ARC_ZE_LOADER_ALLOW_ANY", "0", 1);
    setenv("ARC_ZE_LOADER_PATH", bad_name, 1);
    CHECK(arc_ze_init_ex(&status) == NULL && status == ARC_ZE_ERR_LOAD);
    setenv("ARC_ZE_LOADER_ALLOW_ANY", "true", 1);
    CHECK(arc_ze_init_ex(&status) == NULL && status == ARC_ZE_ERR_LOAD);

    unsetenv("ARC_ZE_LOADER_ALLOW_ANY");
    setenv("ARC_ZE_LOADER_PATH", good, 1);
    ArcZeContext* ctx = arc_ze_init_ex(&status);
    CHECK(ctx != NULL && status == ARC_ZE_OK);
    if (ctx) {
        arc_ze_destroy(ctx);
    }

    setenv("ARC_ZE_LOADER_ALLOW_ANY", "1", 1);
    setenv("ARC_ZE_LOADER_PATH", bad_name, 1);
    ctx = arc_ze_init_ex(&status);
    CHECK(ctx != NULL && status == ARC_ZE_OK);
    if (ctx) {
        arc_ze_destroy(ctx);
    }

    unlink(good);
    unlink(bad_name);
    rmdir(dir_named);
    rmdir(dir);
    setenv("ARC_ZE_LOADER_PATH", fake_path, 1);
}

static void test_alloc_guards_and_tracking(void) {
    reset_env();
    ArcZeContext* ctx = init_ok();
    if (!ctx) {
        return;
    }
    CHECK(arc_ze_alloc_usm(ctx, 0, ARC_USM_HOST) == NULL && arc_ze_last_status(ctx) == ARC_ZE_ERR_INVALID_ARG);
    CHECK(arc_ze_alloc_usm(ctx, (size_t)1 << 62, ARC_USM_HOST) == NULL && arc_ze_last_status(ctx) == ARC_ZE_ERR_TOO_LARGE);
    CHECK(arc_ze_alloc_usm(ctx, (size_t)ARC_MAX_USM_BYTES + 1u, ARC_USM_SHARED) == NULL);
    CHECK(arc_ze_alloc_usm(ctx, 64, (ArcUsmType)7) == NULL && arc_ze_last_status(ctx) == ARC_ZE_ERR_INVALID_ARG);
    CHECK(arc_ze_alloc_usm(NULL, 64, ARC_USM_HOST) == NULL);

    CHECK(arc_ze_alloc_usm(ctx, (size_t)9 << 30, ARC_USM_DEVICE) == NULL && arc_ze_last_status(ctx) == ARC_ZE_ERR_API);
    CHECK(arc_ze_last_result(ctx) != 0u);

    void* host = arc_ze_alloc_usm(ctx, 4096, ARC_USM_HOST);
    void* device = arc_ze_alloc_usm(ctx, 4096, ARC_USM_DEVICE);
    void* shared = arc_ze_alloc_usm(ctx, 4096, ARC_USM_SHARED);
    CHECK(host && device && shared && arc_ze_alloc_count(ctx) == 3u && g_live_allocs() == 3u);

    arc_ze_free_usm(ctx, host);
    CHECK(arc_ze_alloc_count(ctx) == 2u && g_live_allocs() == 2u);
    arc_ze_free_usm(ctx, host);
    CHECK(arc_ze_alloc_count(ctx) == 2u && g_live_allocs() == 2u && arc_ze_last_status(ctx) == ARC_ZE_ERR_INVALID_ARG);
    arc_ze_free_usm(ctx, NULL);
    int on_stack = 0;
    arc_ze_free_usm(ctx, &on_stack);
    CHECK(arc_ze_alloc_count(ctx) == 2u);

    arc_ze_destroy(ctx);
    CHECK(g_live_allocs() == 0u);
    arc_ze_destroy(NULL);
}

static void test_alloc_table_full_frees_the_new_allocation(void) {
    reset_env();
    ArcZeContext* ctx = init_ok();
    if (!ctx) {
        return;
    }
    for (unsigned i = 0; i < ARC_MAX_TRACKED_ALLOCS; ++i) {
        if (!arc_ze_alloc_usm(ctx, 16, ARC_USM_HOST)) {
            CHECK(0);
            break;
        }
    }
    CHECK(arc_ze_alloc_count(ctx) == ARC_MAX_TRACKED_ALLOCS);
    CHECK(arc_ze_alloc_usm(ctx, 16, ARC_USM_HOST) == NULL && arc_ze_last_status(ctx) == ARC_ZE_ERR_TOO_MANY_ALLOCS);
    CHECK(g_live_allocs() == ARC_MAX_TRACKED_ALLOCS);
    arc_ze_destroy(ctx);
    CHECK(g_live_allocs() == 0u);
}

static void test_dma_copy_and_fallback(void) {
    reset_env();
    ArcZeContext* ctx = init_ok();
    if (!ctx) {
        return;
    }
    char* host = (char*)arc_ze_alloc_usm(ctx, 64, ARC_USM_HOST);
    char* shared = (char*)arc_ze_alloc_usm(ctx, 64, ARC_USM_SHARED);
    char* device = (char*)arc_ze_alloc_usm(ctx, 64, ARC_USM_DEVICE);
    if (!host || !shared || !device) {
        CHECK(0);
        arc_ze_destroy(ctx);
        return;
    }
    memset(host, 0x5A, 64);
    memset(shared, 0, 64);

    CHECK(arc_ze_dma_copy_async(ctx, shared, host, 64) == ARC_ZE_OK);
    CHECK(arc_ze_dma_copy_async(ctx, NULL, host, 64) == ARC_ZE_ERR_INVALID_ARG);
    CHECK(arc_ze_dma_copy_async(ctx, shared, host, 0) == ARC_ZE_ERR_INVALID_ARG);
    CHECK(arc_ze_dma_copy_async(NULL, shared, host, 64) == ARC_ZE_ERR_INVALID_ARG);

    setenv("FAKE_ZE_COPY_FAIL", "1", 1);
    /* a queued GPU copy is still pending: a CPU fallback would reorder against it, so it is refused */
    CHECK(arc_ze_dma_copy_async(ctx, shared, host, 64) == ARC_ZE_ERR_API);
    unsetenv("FAKE_ZE_COPY_FAIL");
    CHECK(arc_ze_synchronize(ctx) == ARC_ZE_OK);
    setenv("FAKE_ZE_COPY_FAIL", "1", 1);
    CHECK(arc_ze_dma_copy_async(ctx, shared, host, 64) == ARC_ZE_FALLBACK_MEMCPY);
    CHECK(memcmp(shared, host, 64) == 0);
    CHECK(arc_ze_dma_copy_async(ctx, device, host, 64) == ARC_ZE_ERR_API);
    char untracked[64];
    CHECK(arc_ze_dma_copy_async(ctx, untracked, host, 64) == ARC_ZE_ERR_API);
    /* the whole range must lie inside both allocations (start pointer alone is not enough) */
    CHECK(arc_ze_dma_copy_async(ctx, shared, host, 4096) == ARC_ZE_ERR_API);
    CHECK(arc_ze_dma_copy_async(ctx, shared + 40, host, 32) == ARC_ZE_ERR_API);
    CHECK(arc_ze_dma_copy_async(ctx, shared, host + 40, 32) == ARC_ZE_ERR_API);
    CHECK(arc_ze_dma_copy_async(ctx, shared, host, (size_t)-1) == ARC_ZE_ERR_API);
    reset_env();
    arc_ze_destroy(ctx);
}

static void test_synchronize_is_bounded_and_marks_dead(void) {
    reset_env();
    ArcZeContext* ctx = init_ok();
    if (!ctx) {
        return;
    }
    CHECK(arc_ze_synchronize(ctx) == ARC_ZE_OK && arc_ze_is_dead(ctx) == 0);
    CHECK(arc_ze_synchronize(NULL) == ARC_ZE_ERR_INVALID_ARG);
    arc_ze_destroy(ctx);

    setenv("FAKE_ZE_SYNC", "slow:600", 1);
    ctx = init_ok();
    unsigned long long start = now_ms();
    CHECK(ctx && arc_ze_synchronize(ctx) == ARC_ZE_OK);
    unsigned long long slow_ms = now_ms() - start;
    CHECK(slow_ms >= 550u && slow_ms < ARC_ZE_MAX_SYNC_TIMEOUT_MS);
    arc_ze_destroy(ctx);

    setenv("FAKE_ZE_SYNC", "never", 1);
    setenv("ARC_ZE_SYNC_TIMEOUT_MS", "300", 1);
    ctx = init_ok();
    CHECK(ctx && arc_ze_sync_timeout_ms(ctx) == 300u);
    start = now_ms();
    CHECK(ctx && arc_ze_synchronize(ctx) == ARC_ZE_ERR_TIMEOUT);
    unsigned long long waited = now_ms() - start;
    CHECK(waited >= 290u && waited < 1200u);
    CHECK(ctx && arc_ze_is_dead(ctx) == 1);
    CHECK(ctx && arc_ze_synchronize(ctx) == ARC_ZE_ERR_DEAD);
    CHECK(ctx && arc_ze_alloc_usm(ctx, 64, ARC_USM_HOST) == NULL && arc_ze_last_status(ctx) == ARC_ZE_ERR_DEAD);
    CHECK(ctx && arc_ze_dma_copy_async(ctx, &waited, &start, 8) == ARC_ZE_ERR_DEAD);
    arc_ze_destroy(ctx);

    setenv("FAKE_ZE_SYNC", "lost", 1);
    ctx = init_ok();
    CHECK(ctx && arc_ze_synchronize(ctx) == ARC_ZE_ERR_API && arc_ze_is_dead(ctx) == 1);
    CHECK(ctx && arc_ze_last_result(ctx) == 0x70000001u);
    arc_ze_destroy(ctx);

    setenv("ARC_ZE_SYNC_TIMEOUT_MS", "999999", 1);
    unsetenv("FAKE_ZE_SYNC");
    ctx = init_ok();
    CHECK(ctx && arc_ze_sync_timeout_ms(ctx) == ARC_ZE_MAX_SYNC_TIMEOUT_MS);
    arc_ze_destroy(ctx);
    setenv("ARC_ZE_SYNC_TIMEOUT_MS", "garbage", 1);
    ctx = init_ok();
    CHECK(ctx && arc_ze_sync_timeout_ms(ctx) == ARC_ZE_DEFAULT_SYNC_TIMEOUT_MS);
    arc_ze_destroy(ctx);
    reset_env();
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <fake_loader.so>\n", argv[0]);
        return 2;
    }
    void* fake = dlopen(argv[1], RTLD_NOW);
    if (!fake) {
        fprintf(stderr, "cannot load fake loader: %s\n", dlerror());
        return 2;
    }
    void* symbol = dlsym(fake, "fake_ze_live_allocs");
    if (!symbol) {
        fprintf(stderr, "fake loader lacks fake_ze_live_allocs\n");
        return 2;
    }
    memcpy(&g_live_allocs, &symbol, sizeof symbol);
    setenv("ARC_ZE_LOADER_PATH", argv[1], 1);
    setenv("ARC_ZE_LOADER_ALLOW_ANY", "1", 1);

    test_selects_gpu_not_npu();
    test_init_reports_every_failing_driver_call();
    test_missing_loader_and_symbols();
    test_loader_override_policy(argv[1]);
    setenv("ARC_ZE_LOADER_ALLOW_ANY", "1", 1);
    test_alloc_guards_and_tracking();
    test_alloc_table_full_frees_the_new_allocation();
    test_dma_copy_and_fallback();
    test_synchronize_is_bounded_and_marks_dead();

    printf("checks=%u failures=%u\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
