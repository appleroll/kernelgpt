/*
    MooseOS Kernel
    Copyright (c) 2025 Ethan Zhang
    Licensed under the MIT license. See license file for details
*/
#include <stdint.h>
#include <stddef.h>
#include "vga_text.h"
#include "paging/paging.h"
#include "heap/heap.h"
#include "idt/idt.h"
#include "isr/isr.h"
#include "kernelgpt/kernelgpt.h"
#include "print/debug.h"

void enable_fpu() {
    size_t cr4;
    asm volatile ("mov %%cr4, %0" : "=r"(cr4));
    cr4 |= 0x200; // Set OSFXSR
    cr4 |= 0x400; // Set OSXMMEXCPT
    asm volatile ("mov %0, %%cr4" : : "r"(cr4));
    
    size_t cr0;
    asm volatile ("mov %%cr0, %0" : "=r"(cr0));
    cr0 &= ~(1 << 2); // Clear EM (Emulation)
    cr0 |= (1 << 1);  // Set MP (Monitor Co-processor)
    asm volatile ("mov %0, %%cr0" : : "r"(cr0));
    
    asm volatile ("fninit");
}

void kernel_main(void) 
{
    vga_init();
    vga_puts("[KGPT] Booted.\n");
    
    // Enable FPU for float math
    enable_fpu();
    vga_puts("[KGPT] FPU Enabled.\n");

    paging_init(16 * 1024 * 1024);
    vga_puts("[KGPT] Paging Enabled.\n");

    idt_init();
    isr_init();
    asm volatile("sti"); // Enable interrupts
    vga_puts("[KGPT] Interrupts Enabled.\n");
    
    // Run the GPT
    vga_puts("[KGPT] Starting Training & Inference...\n\n");
    kernelgpt_main();
    
    // Halt
    while(1) {
        asm volatile("hlt");
    }
}
