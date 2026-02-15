#ifndef KERNELGPT_MODEL_H
#define KERNELGPT_MODEL_H

#include "kernelgpt/autograd.h"
#include "kernelgpt/matrix.h"

#define BLOCK_SIZE 16
#define N_EMBD 16
#define N_HEAD 4
#define N_LAYER 1
#define HEAD_DIM (N_EMBD / N_HEAD)

extern int vocab_sz;
extern char* unique_chars;

void init_model(int vocab_size);
void init_optimizer(void);
void optimizer_step(int t);
Value** gpt(int token_id, int pos_id);
int get_token(char c);
char get_char(int t);
Value** softmax(Value** logits, int n);

#endif
