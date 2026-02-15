#ifndef KERNELGPT_MATRIX_H
#define KERNELGPT_MATRIX_H

#include "kernelgpt/autograd.h"

typedef struct {
    int rows;
    int cols;
    Value*** m; // m[rows][cols]
} Matrix;

Matrix matrix_create(int rows, int cols);
void matrix_init_randn(Matrix mat, double std);
void matrix_init_zeros(Matrix mat);
Value** matmul_vec(Matrix W, Value** x);

#endif
