#include "kernelgpt/matrix.h"
#include "kernelgpt/arena.h"
#include "kernelgpt/math_utils.h"
#include "heap/heap.h"

Matrix matrix_create(int rows, int cols) {
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.m = (Value***)kmalloc(rows * sizeof(Value**));
    for(int i=0; i<rows; i++) {
        mat.m[i] = (Value**)kmalloc(cols * sizeof(Value*));
    }
    return mat;
}

void matrix_init_randn(Matrix mat, double std) {
    for(int i=0; i<mat.rows; i++) {
        for(int j=0; j<mat.cols; j++) {
            mat.m[i][j] = param_create(random_gauss(0, std));
        }
    }
}

void matrix_init_zeros(Matrix mat) {
    for(int i=0; i<mat.rows; i++) {
        for(int j=0; j<mat.cols; j++) {
            mat.m[i][j] = param_create(0.0);
        }
    }
}

Value** matmul_vec(Matrix W, Value** x) {
    Value** out = (Value**)arena_alloc(W.rows * sizeof(Value*));
    for(int i=0; i<W.rows; i++) {
        Value* sum = value_create(0.0); // Bias could be added here
        for(int j=0; j<W.cols; j++) {
            sum = add(sum, mul(W.m[i][j], x[j]));
        }
        out[i] = sum;
    }
    return out;
}
