#ifndef VGA_TEXT_H
#define VGA_TEXT_H

#include <stdint.h>
#include <stddef.h>

void vga_init();
void vga_clear();
void vga_putc(char c);
void vga_puts(const char* str);
void vga_printf(const char* fmt, ...);

#endif
