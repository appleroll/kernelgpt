#ifndef KERNELGPT_ARENA_H
#define KERNELGPT_ARENA_H

#include <stddef.h>

void arena_init(void);
void arena_reset(void);
void* arena_alloc(size_t size);

#endif
