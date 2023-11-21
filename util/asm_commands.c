#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>

#include <emmintrin.h>
#include <x86intrin.h>

#include <stdint.h>

// #include "labspectre.h"
// #include "labspectreipc.h"

/*
 * mfence
 * Adds a memory fence
 */
static inline void mfence() {
    asm volatile("mfence");
}

/*
 * clflush
 * Flushes an address from the cache for you
 *
 * Arguments:
 *  - addr: A virtual address whose cache line we should flush
 *
 * Returns: None
 * Side Effects: Flushes a cache line from the cache
 */
void clflush(void *addr) {
    _mm_clflush(addr);
}

/*
 * rdtsc
 * Reads the current timestamp counter
 *
 * Returns: Current value of TSC
 */
uint64_t rdtsc(void) {
    return __rdtsc();
}

/*
 * time_access
 * Returns the time to access an address
 */
uint64_t time_access(void *addr) {
    unsigned int tmp;
    uint64_t time1, time2;
    time1 = __rdtscp(&tmp);
    tmp = *(unsigned int *)addr;
    time2 = __rdtscp(&tmp);
    return time2 - time1;
}