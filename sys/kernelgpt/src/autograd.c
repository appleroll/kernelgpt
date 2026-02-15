#include "kernelgpt/autograd.h"
#include "kernelgpt/arena.h"
#include "heap/heap.h"
#include "math.h"
#include "vga_text.h"

Value* value_create(double data) {
    Value* v = (Value*)arena_alloc(sizeof(Value));
    v->data = data;
    v->grad = 0;
    v->num_children = 0;
    v->children[0] = NULL; v->children[1] = NULL;
    v->visited = 0;
    return v;
}

Value* param_create(double data) {
    Value* v = (Value*)kmalloc(sizeof(Value)); // Persist on heap
    if (!v) {
        vga_puts("PANIC: Failed to allocate Parameter!\n");
        while(1);
    }
    v->data = data;
    v->grad = 0;
    v->num_children = 0;
    v->children[0] = NULL; v->children[1] = NULL;
    v->visited = 0;
    return v;
}

Value* add(Value* a, Value* b) {
    Value* v = value_create(a->data + b->data);
    v->children[0] = a; v->local_grads[0] = 1.0;
    v->children[1] = b; v->local_grads[1] = 1.0;
    v->num_children = 2;
    return v;
}

Value* mul(Value* a, Value* b) {
    Value* v = value_create(a->data * b->data);
    v->children[0] = a; v->local_grads[0] = b->data;
    v->children[1] = b; v->local_grads[1] = a->data;
    v->num_children = 2;
    return v;
}

Value* power(Value* a, double p) {
    Value* v = value_create(pow(a->data, p));
    v->children[0] = a; 
    v->local_grads[0] = p * pow(a->data, p - 1);
    v->num_children = 1;
    return v;
}

Value* v_log(Value* a) {
    Value* v = value_create(log(a->data));
    v->children[0] = a;
    v->local_grads[0] = 1.0 / a->data;
    v->num_children = 1;
    return v;
}

Value* v_exp(Value* a) {
    Value* v = value_create(exp(a->data));
    v->children[0] = a;
    v->local_grads[0] = v->data; // exp(x)' = exp(x)
    v->num_children = 1;
    return v;
}

Value* relu(Value* a) {
    Value* v = value_create(a->data > 0 ? a->data : 0);
    v->children[0] = a;
    v->local_grads[0] = (a->data > 0) ? 1.0 : 0.0;
    v->num_children = 1;
    return v;
}

// Topological Sort & Backward
#define MAX_NODES 100000 
static Value* topo[MAX_NODES];
static int topo_idx = 0;

void build_topo(Value* v) {
    if (v->visited) return;
    v->visited = 1;
    for (int i = 0; i < v->num_children; i++) {
        build_topo(v->children[i]);
    }
    if (topo_idx < MAX_NODES)
        topo[topo_idx++] = v;
}

void backward(Value* root) {
    topo_idx = 0;
    
    build_topo(root);
    
    root->grad = 1.0;
    
    for (int i = topo_idx - 1; i >= 0; i--) {
        Value* v = topo[i];
        for (int j = 0; j < v->num_children; j++) {
            Value* child = v->children[j];
            child->grad += v->local_grads[j] * v->grad;
        }
    }
    
    // Reset visited for next pass
    for (int i = 0; i < topo_idx; i++) {
        topo[i]->visited = 0;
    }
}
