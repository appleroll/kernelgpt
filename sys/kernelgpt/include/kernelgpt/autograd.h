#ifndef KERNELGPT_AUTOGRAD_H
#define KERNELGPT_AUTOGRAD_H

typedef struct Value Value;
struct Value {
    double data;
    double grad;
    Value* children[2]; // Max 2 children for binary ops
    int num_children;
    double local_grads[2]; 
    // To support topological sort easily:
    int visited; 
};

Value* value_create(double data);
Value* param_create(double data);
Value* add(Value* a, Value* b);
Value* mul(Value* a, Value* b);
Value* power(Value* a, double p);
Value* v_log(Value* a);
Value* v_exp(Value* a);
Value* relu(Value* a);

void backward(Value* root);

#endif
