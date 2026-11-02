/**
 * System Governor for Intel Arc Architecture.
 *
 * Implements Win32 process priority boosting, CPU core affinity pinning to
 * Performance cores (P-cores), and multimedia timer resolution control.
 */

#ifndef ARC_GOVERNOR_H
#define ARC_GOVERNOR_H

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
    uint32_t original_priority_class;
    uint64_t original_affinity_mask;
    uint32_t timer_period_active;
    uint32_t status_flags;
} arc_governor_state_t;

/**
 * Initialize governor state structure and capture current OS process parameters.
 *
 * @param state Pointer to arc_governor_state_t struct.
 * @return 0 on success, non-zero on error.
 */
ARC_API int arc_governor_init(arc_governor_state_t* state);

/**
 * Boost current process to HIGH_PRIORITY_CLASS (0x00000080).
 *
 * @param state Pointer to active state struct for tracking original priority.
 * @return 0 on success, non-zero on error.
 */
ARC_API int arc_governor_boost_high(arc_governor_state_t* state);

/**
 * Set process affinity mask to pin execution to specified CPU cores (e.g. P-cores).
 *
 * @param state Pointer to active state struct.
 * @param core_mask 64-bit affinity bitmask.
 * @return 0 on success, non-zero on error.
 */
ARC_API int arc_governor_set_affinity(arc_governor_state_t* state, uint64_t core_mask);

/**
 * Enable 1ms multimedia high-resolution timer.
 *
 * @param state Pointer to active state struct.
 * @return 0 on success, non-zero on error.
 */
ARC_API int arc_governor_enable_timer(arc_governor_state_t* state);

/**
 * Restore original OS process priority, CPU affinity mask, and timer period.
 *
 * @param state Pointer to state struct containing recorded original settings.
 * @return 0 on success, non-zero on error.
 */
ARC_API int arc_governor_restore(arc_governor_state_t* state);

#ifdef __cplusplus
}
#endif

#endif /* ARC_GOVERNOR_H */
