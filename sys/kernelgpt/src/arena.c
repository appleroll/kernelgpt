#include "kernelgpt/arena.h"
#include "heap/heap.h"
#include "vga_text.h"

// Arena allocator for autograd graph nodes - we will reset it after each training iteration
#define ARENA_SIZE (8 * 1024 * 1024) // 8MB arena
static char* arena_buffer = NULL;
static size_t arena_offset = 0;

void arena_init(void) {
    arena_buffer = (char*)kmalloc(ARENA_SIZE);
    if (!arena_buffer) {
        vga_puts("PANIC: Failed to allocate Arena!\n");
        while(1);
    }
    arena_offset = 0;
    vga_puts("Arena allocated.\n");
}

void arena_reset(void) {
    arena_offset = 0;
}

void* arena_alloc(size_t size) {
    // Align to 8 bytes
    if (arena_offset % 8 != 0) arena_offset += (8 - (arena_offset % 8));
    if (arena_offset + size > ARENA_SIZE) {
        vga_puts("PANIC: Arena Out of Memory!\n");
        while(1);
    }
    void* ptr = &arena_buffer[arena_offset];
    arena_offset += size;
    return ptr;
}
