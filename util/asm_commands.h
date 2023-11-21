#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>

#include <emmintrin.h>
#include <x86intrin.h>

void mfence(void);
u_int64_t clflush(void*);
u_int64_t rdtsc(void);
u_int64_t time_access(void*);