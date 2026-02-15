#include "vga_text.h"
#include <stdarg.h>
#include "print/debug.h"

#define VGA_WIDTH 80
#define VGA_HEIGHT 25
#define VGA_MEMORY 0xB8000

static uint16_t* vga_buffer = (uint16_t*)VGA_MEMORY;
static int terminal_row = 0;
static int terminal_col = 0;
static uint8_t terminal_color = 0x07; // Light grey on black

static uint16_t vga_entry(unsigned char uc, uint8_t color) {
    return (uint16_t) uc | (uint16_t) color << 8;
}

void vga_init() {
    terminal_row = 0;
    terminal_col = 0;
    terminal_color = 0x07;
    vga_clear();
}

void vga_clear() {
    for (int y = 0; y < VGA_HEIGHT; y++) {
        for (int x = 0; x < VGA_WIDTH; x++) {
            const size_t index = y * VGA_WIDTH + x;
            vga_buffer[index] = vga_entry(' ', terminal_color);
        }
    }
    terminal_row = 0;
    terminal_col = 0;
}

void vga_scroll() {
    for (int y = 0; y < VGA_HEIGHT - 1; y++) {
        for (int x = 0; x < VGA_WIDTH; x++) {
            vga_buffer[y * VGA_WIDTH + x] = vga_buffer[(y + 1) * VGA_WIDTH + x];
        }
    }
    for (int x = 0; x < VGA_WIDTH; x++) {
        vga_buffer[(VGA_HEIGHT - 1) * VGA_WIDTH + x] = vga_entry(' ', terminal_color);
    }
    terminal_row = VGA_HEIGHT - 1;
}

void vga_putc(char c) {
    // Send to QEMU serial console for debugging
    char buf[2] = {c, 0};
    debugf(buf);

    if (c == '\n') {
        terminal_col = 0;
        terminal_row++;
    } else {
        const size_t index = terminal_row * VGA_WIDTH + terminal_col;
        vga_buffer[index] = vga_entry(c, terminal_color);
        terminal_col++;
    }

    if (terminal_col >= VGA_WIDTH) {
        terminal_col = 0;
        terminal_row++;
    }

    if (terminal_row >= VGA_HEIGHT) {
        vga_scroll(); // Scroll one line
    }
}

void vga_puts(const char* str) {
    for (size_t i = 0; str[i] != '\0'; i++) {
        vga_putc(str[i]);
    }
}

// Helpers
static void print_int(int num) {
    char buf[32];
    int i = 0;
    if (num == 0) {
        vga_putc('0');
        return;
    }
    int sign = 0;
    if (num < 0) {
        sign = 1;
        num = -num;
    }
    while (num > 0) {
        buf[i++] = (num % 10) + '0';
        num /= 10;
    }
    if (sign) vga_putc('-');
    while (--i >= 0) vga_putc(buf[i]);
}

void vga_printf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);

    for (const char* p = fmt; *p != '\0'; p++) {
        if (*p != '%') {
            vga_putc(*p);
            continue;
        }
        
        p++;
        switch (*p) {
            case 'd': {
                int i = va_arg(args, int);
                print_int(i);
                break;
            }
            case 's': {
                char* s = va_arg(args, char*);
                vga_puts(s ? s : "(null)");
                break;
            }
            case 'c': {
                int c = va_arg(args, int);
                vga_putc(c);
                break;
            }
            case 'f': {
                // Approximate float printing
                double d = va_arg(args, double);
                if (d < 0) {
                    vga_putc('-');
                    d = -d;
                }
                int i = (int)d;
                print_int(i);
                vga_putc('.');
                d -= i;
                d *= 10000; // 4 decimal places
                int f = (int)d;
                if (f < 1000) vga_putc('0');
                if (f < 100) vga_putc('0');
                if (f < 10) vga_putc('0');
                print_int(f);
                break;
            }
            default:
                vga_putc('%');
                vga_putc(*p);
                break;
        }
    }
    va_end(args);
}
