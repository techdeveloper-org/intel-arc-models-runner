/**
 * System Governor implementation for Win32 Intel Core Ultra / Arc platforms.
 */

#include "arc_governor.h"

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <timeapi.h>
#endif

int arc_governor_init(arc_governor_state_t* state) {
    if (!state) return -1;
#ifdef _WIN32
    HANDLE hProcess = GetCurrentProcess();
    state->original_priority_class = GetPriorityClass(hProcess);
    
    DWORD_PTR procMask = 0;
    DWORD_PTR sysMask = 0;
    if (GetProcessAffinityMask(hProcess, &procMask, &sysMask)) {
        state->original_affinity_mask = (uint64_t)procMask;
    } else {
        state->original_affinity_mask = 0;
    }
    state->timer_period_active = 0;
    state->status_flags = 1;
    return 0;
#else
    state->original_priority_class = 0;
    state->original_affinity_mask = 0;
    state->timer_period_active = 0;
    state->status_flags = 0;
    return 0;
#endif
}

int arc_governor_boost_high(arc_governor_state_t* state) {
#ifdef _WIN32
    HANDLE hProcess = GetCurrentProcess();
    if (state && state->original_priority_class == 0) {
        state->original_priority_class = GetPriorityClass(hProcess);
    }
    // HIGH_PRIORITY_CLASS = 0x00000080
    if (!SetPriorityClass(hProcess, HIGH_PRIORITY_CLASS)) {
        return (int)GetLastError();
    }
    return 0;
#else
    (void)state;
    return 0;
#endif
}

int arc_governor_set_affinity(arc_governor_state_t* state, uint64_t core_mask) {
    if (core_mask == 0) return -1;
#ifdef _WIN32
    (void)state;
    HANDLE hProcess = GetCurrentProcess();
    if (!SetProcessAffinityMask(hProcess, (DWORD_PTR)core_mask)) {
        return (int)GetLastError();
    }
    return 0;
#else
    (void)state;
    (void)core_mask;
    return 0;
#endif
}

int arc_governor_enable_timer(arc_governor_state_t* state) {
#ifdef _WIN32
    if (timeBeginPeriod(1) == TIMERR_NOERROR) {
        if (state) state->timer_period_active = 1;
        return 0;
    }
    return -1;
#else
    (void)state;
    return 0;
#endif
}

int arc_governor_restore(arc_governor_state_t* state) {
    if (!state) return -1;
#ifdef _WIN32
    HANDLE hProcess = GetCurrentProcess();
    if (state->original_priority_class != 0) {
        SetPriorityClass(hProcess, state->original_priority_class);
    }
    if (state->original_affinity_mask != 0) {
        SetProcessAffinityMask(hProcess, (DWORD_PTR)state->original_affinity_mask);
    }
    if (state->timer_period_active) {
        timeEndPeriod(1);
        state->timer_period_active = 0;
    }
    return 0;
#else
    return 0;
#endif
}
