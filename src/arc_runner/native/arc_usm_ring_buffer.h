/**
 * Lock-Free Single-Producer Single-Consumer (SPSC) Circular Ring Buffer
 * for Level Zero USM Token Streaming.
 */

#ifndef ARC_USM_RING_BUFFER_H
#define ARC_USM_RING_BUFFER_H

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
  #ifdef ARC_EXPORTS
    #define ARC_API __declspec(dllexport)
  #else
    #define ARC_API __declspec(dllimport)
  #endif
#else
  #define ARC_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t head;            /* Consumer read index (64-byte cacheline aligned) */
    uint8_t pad0[60];
    uint32_t tail;            /* Producer write index (64-byte cacheline aligned) */
    uint8_t pad1[60];
    uint32_t capacity;        /* Buffer power-of-two capacity */
    uint32_t mask;            /* capacity - 1 */
    int64_t* tokens;          /* Token identifier storage buffer */
    float* latencies_ms;      /* Per-token generation latency deltas */
} arc_usm_ring_buffer_t;

/**
 * Allocate and initialize a lock-free circular ring buffer.
 *
 * @param capacity Number of tokens the buffer can hold (will round up to power-of-two).
 * @return Pointer to allocated arc_usm_ring_buffer_t, or NULL on failure.
 */
ARC_API arc_usm_ring_buffer_t* arc_usm_ring_create(size_t capacity);

/**
 * Push a generated token into the ring buffer (Producer).
 *
 * @param ring Active ring buffer pointer.
 * @param token_id Generated token ID.
 * @param latency_ms Latency delta in milliseconds.
 * @return 0 on success, -1 if buffer is full (overflow).
 */
ARC_API int arc_usm_ring_push(arc_usm_ring_buffer_t* ring, int64_t token_id, float latency_ms);

/**
 * Pop the next token from the ring buffer (Consumer).
 *
 * @param ring Active ring buffer pointer.
 * @param out_token_id Pointer to receive the token ID.
 * @param out_latency_ms Pointer to receive the latency delta.
 * @return 0 on success, -1 if buffer is empty.
 */
ARC_API int arc_usm_ring_pop(arc_usm_ring_buffer_t* ring, int64_t* out_token_id, float* out_latency_ms);

/**
 * Deallocate ring buffer memory.
 *
 * @param ring Ring buffer to destroy.
 */
ARC_API void arc_usm_ring_destroy(arc_usm_ring_buffer_t* ring);

#ifdef __cplusplus
}
#endif

#endif /* ARC_USM_RING_BUFFER_H */
