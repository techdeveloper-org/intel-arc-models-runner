/**
 * @file mmap_harness.c
 * @brief Sanitizer-backed tests for the bounds checks in arc_flash_stream_mmap.c.
 *
 * Build: gcc -std=c17 -fsanitize=address,undefined -fno-sanitize-recover=all
 *        mmap_harness.c ../../src/arc_runner/native/arc_flash_stream_mmap.c
 * Usage: mmap_harness <scratch_dir>
 */

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "../../src/arc_runner/native/arc_flash_stream_mmap.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FILE_BYTES 4096u
#define FUZZ_ITERATIONS 1000000u

static unsigned g_failures = 0;
static unsigned g_checks = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failures;                                                        \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
        }                                                                        \
    } while (0)

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;

/**
 * @brief xorshift64* pseudo-random generator, deterministic for reproducible fuzzing.
 *
 * @return Next 64-bit pseudo-random value.
 */
static uint64_t next_random(void) {
    g_rng ^= g_rng >> 12;
    g_rng ^= g_rng << 25;
    g_rng ^= g_rng >> 27;
    return g_rng * 0x2545F4914F6CDD1Dull;
}

/**
 * @brief Pick a value biased toward the boundaries that break naive arithmetic.
 *
 * @param fsize File size used to place near-boundary values.
 * @return Candidate offset or size.
 */
static uint64_t edge_biased(uint64_t fsize) {
    static const uint64_t fixed[] = {0u, 1u, 2u, UINT64_MAX, UINT64_MAX - 1u, UINT64_MAX - 7u, 1ull << 63, (1ull << 63) - 1u, 1ull << 32};
    uint64_t r = next_random();
    switch (r % 5u) {
        case 0: return fixed[(r >> 8) % (sizeof fixed / sizeof fixed[0])];
        case 1: return fsize + (r >> 8) % 4u - 2u;
        case 2: return fsize == UINT64_MAX ? (r >> 8) : (r >> 8) % (fsize + 1u);
        case 3: return fsize - (r >> 8) % 3u;
        default: return next_random();
    }
}

/**
 * @brief Test arc_range_ok against an add-with-carry reference on boundaries and random input.
 */
static void test_range_ok_matches_carry_reference(void) {
    const uint64_t sizes[] = {1u, FILE_BYTES, 1ull << 31, 1ull << 40, UINT64_MAX};
    for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; ++s) {
        for (unsigned i = 0; i < FUZZ_ITERATIONS / 5u; ++i) {
            uint64_t fsize = sizes[s];
            uint64_t off = edge_biased(fsize);
            uint64_t sz = edge_biased(fsize);
            uint64_t end = off + sz;
            bool carried = end < off;
            bool expected = !carried && end <= fsize;
            if (arc_range_ok(fsize, off, sz) != expected) {
                ++g_failures;
                fprintf(stderr, "FAIL range fsize=%" PRIu64 " off=%" PRIu64 " sz=%" PRIu64 "\n", fsize, off, sz);
                return;
            }
            ++g_checks;
        }
    }
    CHECK(!arc_range_ok(FILE_BYTES, UINT64_MAX - 3u, 8u));
    CHECK(!arc_range_ok(FILE_BYTES, 0u, UINT64_MAX));
    CHECK(!arc_range_ok(FILE_BYTES, FILE_BYTES - 6u, 7u));
    CHECK(arc_range_ok(FILE_BYTES, FILE_BYTES - 6u, 6u));
    CHECK(arc_range_ok(FILE_BYTES, FILE_BYTES, 0u));
    CHECK(!arc_range_ok(FILE_BYTES, FILE_BYTES + 1u, 0u));
}

/**
 * @brief Write the patterned test file used by the mapping tests.
 *
 * @param path Destination path.
 * @param bytes Number of bytes to write; byte i holds i & 0xFF.
 * @return 0 on success, non-zero on failure.
 */
static int write_pattern_file(const char* path, unsigned bytes) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        return 1;
    }
    for (unsigned i = 0; i < bytes; ++i) {
        fputc((int)(i & 0xFFu), f);
    }
    return fclose(f);
}

/**
 * @brief Fill a descriptor with one named tensor.
 *
 * @param layer Descriptor to initialize (zeroed first).
 * @param idx Layer index.
 * @param name Tensor name.
 * @param off File offset.
 * @param size Payload size.
 */
static void make_layer(ArcLayerDescriptor* layer, int32_t idx, const char* name, uint64_t off, uint64_t size) {
    memset(layer, 0, sizeof *layer);
    layer->layer_idx = idx;
    layer->tensor_count = 1;
    snprintf(layer->tensors[0].name, ARC_MAX_NAME_LEN, "%s", name);
    layer->tensors[0].n_dims = 1;
    layer->tensors[0].dims[0] = size;
    layer->tensors[0].file_offset = off;
    layer->tensors[0].size_bytes = size;
}

/**
 * @brief Exercise open failures: missing file, empty file, NULL and empty paths.
 *
 * @param dir Scratch directory.
 */
static void test_open_failures(const char* dir) {
    char path[1024];
    int32_t status = 0;

    CHECK(arc_mmap_open(NULL, &status) == NULL && status == ARC_ERR_INVALID_ARG);
    CHECK(arc_mmap_open("", &status) == NULL && status == ARC_ERR_INVALID_ARG);

    snprintf(path, sizeof path, "%s/does_not_exist.bin", dir);
    CHECK(arc_mmap_open(path, &status) == NULL && status == ARC_ERR_OPEN_FAILED);

    snprintf(path, sizeof path, "%s/empty.bin", dir);
    CHECK(write_pattern_file(path, 0u) == 0);
    CHECK(arc_mmap_open(path, &status) == NULL && status == ARC_ERR_OPEN_FAILED);
    CHECK(arc_mmap_open(path, NULL) == NULL);
}

/**
 * @brief Every extent that overflows or leaves the file must be rejected at register time.
 *
 * @param ctx Open context over a FILE_BYTES file.
 */
static void test_register_rejects_bad_extents(ArcMmapContext* ctx) {
    ArcLayerDescriptor* layer = (ArcLayerDescriptor*)calloc(1, sizeof *layer);
    CHECK(layer != NULL);
    if (!layer) {
        return;
    }

    make_layer(layer, 0, "wrap", UINT64_MAX - 3u, 8u);
    CHECK(arc_mmap_register_layer(ctx, layer) == ARC_ERR_BAD_EXTENT);
    make_layer(layer, 0, "huge", 0u, UINT64_MAX);
    CHECK(arc_mmap_register_layer(ctx, layer) == ARC_ERR_BAD_EXTENT);
    make_layer(layer, 0, "past", FILE_BYTES - 6u, 7u);
    CHECK(arc_mmap_register_layer(ctx, layer) == ARC_ERR_BAD_EXTENT);
    make_layer(layer, 0, "start_past_end", FILE_BYTES + 1u, 0u);
    CHECK(arc_mmap_register_layer(ctx, layer) == ARC_ERR_BAD_EXTENT);
    CHECK(arc_mmap_layer_count(ctx) == 0u);

    make_layer(layer, -1, "neg", 0u, 4u);
    CHECK(arc_mmap_register_layer(ctx, layer) == ARC_ERR_INVALID_ARG);
    make_layer(layer, ARC_MAX_LAYERS, "high", 0u, 4u);
    CHECK(arc_mmap_register_layer(ctx, layer) == ARC_ERR_INVALID_ARG);

    make_layer(layer, 0, "dims", 0u, 4u);
    layer->tensors[0].n_dims = 5u;
    CHECK(arc_mmap_register_layer(ctx, layer) == ARC_ERR_INVALID_ARG);

    make_layer(layer, 0, "count", 0u, 4u);
    layer->tensor_count = ARC_MAX_TENSORS_PER_LAYER + 1u;
    CHECK(arc_mmap_register_layer(ctx, layer) == ARC_ERR_TOO_MANY_TENSORS);
    layer->tensor_count = UINT32_MAX;
    CHECK(arc_mmap_register_layer(ctx, layer) == ARC_ERR_TOO_MANY_TENSORS);

    CHECK(arc_mmap_register_layer(NULL, layer) == ARC_ERR_INVALID_ARG);
    CHECK(arc_mmap_register_layer(ctx, NULL) == ARC_ERR_INVALID_ARG);
    CHECK(arc_mmap_layer_count(ctx) == 0u);
    free(layer);
}

/**
 * @brief Valid registrations resolve to the exact mapped bytes; unsealed reads fail.
 *
 * @param ctx Open context over a FILE_BYTES patterned file.
 */
static void test_seal_and_read(ArcMmapContext* ctx) {
    ArcLayerDescriptor* layer = (ArcLayerDescriptor*)calloc(1, sizeof *layer);
    CHECK(layer != NULL);
    if (!layer) {
        return;
    }
    uint64_t size = 0;

    make_layer(layer, 3, "edge", FILE_BYTES - 6u, 6u);
    CHECK(arc_mmap_register_layer(ctx, layer) == ARC_OK);
    CHECK(arc_mmap_get_tensor_ptr(ctx, 3, "edge", &size) == NULL);
    CHECK(arc_mmap_last_status(ctx) == ARC_ERR_NOT_SEALED);

    make_layer(layer, 3, "edge", 10u, 16u);
    CHECK(arc_mmap_register_layer(ctx, layer) == ARC_OK);
    CHECK(arc_mmap_layer_count(ctx) == 1u);

    char long_name[ARC_MAX_NAME_LEN];
    memset(long_name, 'A', sizeof long_name);
    make_layer(layer, 4, "x", 100u, 8u);
    memcpy(layer->tensors[0].name, long_name, sizeof long_name);
    CHECK(arc_mmap_register_layer(ctx, layer) == ARC_OK);

    CHECK(arc_mmap_is_sealed(ctx) == 0);
    CHECK(arc_mmap_seal(ctx) == ARC_OK);
    CHECK(arc_mmap_seal(ctx) == ARC_OK);
    CHECK(arc_mmap_is_sealed(ctx) == 1);
    CHECK(arc_mmap_register_layer(ctx, layer) == ARC_ERR_SEALED);

    const uint8_t* p = arc_mmap_get_tensor_ptr(ctx, 3, "edge", &size);
    CHECK(p != NULL && size == 16u);
    for (unsigned i = 0; p && i < 16u; ++i) {
        CHECK(p[i] == (uint8_t)((10u + i) & 0xFFu));
    }

    char terminated[ARC_MAX_NAME_LEN];
    memset(terminated, 'A', ARC_MAX_NAME_LEN - 1u);
    terminated[ARC_MAX_NAME_LEN - 1u] = '\0';
    p = arc_mmap_get_tensor_ptr(ctx, 4, terminated, &size);
    CHECK(p != NULL && size == 8u);

    CHECK(arc_mmap_get_tensor_ptr(ctx, 3, "missing", &size) == NULL);
    CHECK(arc_mmap_last_status(ctx) == ARC_ERR_NOT_FOUND);
    CHECK(arc_mmap_get_tensor_ptr(ctx, 9, "edge", &size) == NULL);
    CHECK(arc_mmap_get_tensor_ptr(ctx, 3, NULL, &size) == NULL);
    CHECK(arc_mmap_get_tensor_ptr(NULL, 3, "edge", &size) == NULL);
    CHECK(arc_mmap_get_tensor_ptr(ctx, 3, "edge", NULL) != NULL);
    free(layer);
}

/**
 * @brief Lock and unlock must reject wrapped, empty and out-of-file ranges.
 *
 * @param ctx Open context over a FILE_BYTES file.
 */
static void test_lock_range_validation(ArcMmapContext* ctx) {
    CHECK(arc_mmap_lock_range(ctx, 0u, 0u) == ARC_ERR_INVALID_ARG);
    CHECK(arc_mmap_lock_range(ctx, UINT64_MAX - 3u, 8u) == ARC_ERR_RANGE);
    CHECK(arc_mmap_lock_range(ctx, 0u, UINT64_MAX) == ARC_ERR_RANGE);
    CHECK(arc_mmap_lock_range(ctx, FILE_BYTES, 1u) == ARC_ERR_RANGE);
    CHECK(arc_mmap_unlock_range(ctx, UINT64_MAX - 3u, 8u) == ARC_ERR_RANGE);
    CHECK(arc_mmap_lock_range(NULL, 0u, 1u) == ARC_ERR_INVALID_ARG);
    CHECK(arc_mmap_locked_bytes(ctx) == 0u);

    int rc = arc_mmap_lock_range(ctx, 0u, FILE_BYTES);
    CHECK(rc == ARC_OK || rc == ARC_ERR_LOCK_FAILED);
    if (rc == ARC_OK) {
        CHECK(arc_mmap_locked_bytes(ctx) == FILE_BYTES);
        CHECK(arc_mmap_unlock_range(ctx, 0u, FILE_BYTES) == ARC_OK);
        CHECK(arc_mmap_locked_bytes(ctx) == 0u);
    } else {
        CHECK(arc_mmap_last_os_error(ctx) != 0u);
        CHECK(arc_mmap_locked_bytes(ctx) == 0u);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <scratch_dir>\n", argv[0]);
        return 2;
    }
    const char* dir = argv[1];
    char path[1024];

    CHECK(arc_native_abi_version() == ARC_NATIVE_ABI_VERSION);
    CHECK(arc_native_build_info() != NULL);
    arc_mmap_close(NULL);

    test_range_ok_matches_carry_reference();
    test_open_failures(dir);

    snprintf(path, sizeof path, "%s/blob.bin", dir);
    CHECK(write_pattern_file(path, FILE_BYTES) == 0);
    int32_t status = -99;
    ArcMmapContext* ctx = arc_mmap_open(path, &status);
    CHECK(ctx != NULL && status == ARC_OK);
    if (ctx) {
        CHECK(arc_mmap_file_size(ctx) == FILE_BYTES);
        test_register_rejects_bad_extents(ctx);
        test_seal_and_read(ctx);
        test_lock_range_validation(ctx);
        arc_mmap_close(ctx);
    }

    printf("checks=%u failures=%u\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
