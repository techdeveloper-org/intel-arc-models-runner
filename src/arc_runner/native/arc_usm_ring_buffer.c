/**
 * Lock-Free SPSC Ring Buffer implementation.
 */

#include "arc_usm_ring_buffer.h"
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
  #include <windows.h>
  #define MEMORY_BARRIER() MemoryBarrier()
#else
  #define MEMORY_BARRIER() __sync_synchronize()
#endif

static size_t next_power_of_two(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
#if defined(_WIN64) || defined(__x86_64__)
    n |= n >> 32;
#endif
    return n + 1;
}

arc_usm_ring_buffer_t* arc_usm_ring_create(size_t capacity) {
    if (capacity < 16) capacity = 16;
    size_t pow2_cap = next_power_of_two(capacity);

    arc_usm_ring_buffer_t* ring = (arc_usm_ring_buffer_t*)malloc(sizeof(arc_usm_ring_buffer_t));
    if (!ring) return NULL;

    memset(ring, 0, sizeof(arc_usm_ring_buffer_t));
    ring->capacity = (uint32_t)pow2_cap;
    ring->mask = (uint32_t)(pow2_cap - 1);
    ring->head = 0;
    ring->tail = 0;

    ring->tokens = (int64_t*)malloc(sizeof(int64_t) * pow2_cap);
    ring->latencies_ms = (float*)malloc(sizeof(float) * pow2_cap);

    if (!ring->tokens || !ring->latencies_ms) {
        if (ring->tokens) free(ring->tokens);
        if (ring->latencies_ms) free(ring->latencies_ms);
        free(ring);
        return NULL;
    }

    return ring;
}

int arc_usm_ring_push(arc_usm_ring_buffer_t* ring, int64_t token_id, float latency_ms) {
    if (!ring) return -1;

    uint32_t current_tail = ring->tail;
    uint32_t current_head = ring->head;

    /* Check if buffer is full */
    if ((current_tail - current_head) >= ring->capacity) {
        return -1;
    }

    uint32_t idx = current_tail & ring->mask;
    ring->tokens[idx] = token_id;
    ring->latencies_ms[idx] = latency_ms;

    MEMORY_BARRIER();
    ring->tail = current_tail + 1;
    return 0;
}

int arc_usm_ring_pop(arc_usm_ring_buffer_t* ring, int64_t* out_token_id, float* out_latency_ms) {
    if (!ring || !out_token_id || !out_latency_ms) return -1;

    uint32_t current_head = ring->head;
    uint32_t current_tail = ring->tail;

    /* Check if buffer is empty */
    if (current_head == current_tail) {
        return -1;
    }

    uint32_t idx = current_head & ring->mask;
    *out_token_id = ring->tokens[idx];
    *out_latency_ms = ring->latencies_ms[idx];

    MEMORY_BARRIER();
    ring->head = current_head + 1;
    return 0;
}

void arc_usm_ring_destroy(arc_usm_ring_buffer_t* ring) {
    if (!ring) return;
    if (ring->tokens) {
        free(ring->tokens);
        ring->tokens = NULL;
    }
    if (ring->latencies_ms) {
        free(ring->latencies_ms);
        ring->latencies_ms = NULL;
    }
    free(ring);
}
