/*
    MooseOS Panic code
    Copyright (c) 2025 Ethan Zhang
    Licensed under the MIT license. See license file for details
*/

#include "panic/panic.h"
#include "vga_text.h"
#include "print/debug.h"

void panic(const char* message) {
	asm volatile("cli");
	
	vga_puts("\nKERNEL PANIC: ");
	vga_puts(message);
	vga_puts("\nSystem halted.");
	debugf("[MOOSE] PANIC!\n");
	
	// halt the CPU
	while (1) {
		asm volatile("hlt");
	}
}