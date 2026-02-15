/*
    MooseOS Interrupt Descriptor Table code
    Copyright (c) 2025 Ethan Zhang
    Licensed under the MIT license. See license file for details
*/

#include "idt/idt.h"
#include "irq/irq.h"


struct idt_descriptor_t idt_descriptor;

// IDT entry table
struct IDT_entry IDT[IDT_SIZE];

// set an entry in the IDT
void idt_set_entry(unsigned char num, unsigned long base, unsigned short selector, unsigned char flags)
{
    IDT[num].offset_lowerbits = base & 0xFFFF;
    IDT[num].selector = selector;
    IDT[num].zero = 0;
    IDT[num].type_attr = flags;
    IDT[num].offset_higherbits = (base >> 16) & 0xFFFF;
}

// initialise all entries in the IDT
/**
 * @todo add more entries once we get more interrupts (syscalls etc.)
 */
static void initialise_all_entries(void)
{
    // No IRQ handlers for now
}
/**
 * initialize the Interrupt Descriptor Table
 */
void idt_init(void)
{
    gdt_init();
    initialise_all_entries();
    irq_remap();

    idt_descriptor.limit = sizeof(struct IDT_entry) * IDT_SIZE - 1;
    idt_descriptor.base = (unsigned int)IDT;

    idt_load(&idt_descriptor);
}

