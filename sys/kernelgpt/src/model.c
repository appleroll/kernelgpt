#include "kernelgpt/model.h"
#include "kernelgpt/arena.h"
#include "kernelgpt/math_utils.h"
#include "kernelgpt/matrix.h"
#include "kernelgpt/config.h"
#include "vga_text.h"
#include "heap/heap.h"
#include "math.h"
#include "print/debug.h"

// Globals
int vocab_sz = 65;
char* unique_chars = NULL;

static Matrix wte;
static Matrix wpe;
static Matrix lm_head;
static Matrix attn_wq, attn_wk, attn_wv, attn_wo;
static Matrix mlp_fc1, mlp_fc2;

// To collect all params for optimizer
#define MAX_PARAMS 10000
static Value* all_params[MAX_PARAMS];
static int num_params = 0;

static Value*** keys_cache; // [BLOCK_SIZE][N_EMBD]
static Value*** values_cache;

void register_matrix(Matrix m) {
    for(int i=0; i<m.rows; i++) {
        for(int j=0; j<m.cols; j++) {
            if(num_params < MAX_PARAMS)
                all_params[num_params++] = m.m[i][j];
        }
    }
}

void init_model(int vocab_size) {
    wte = matrix_create(vocab_size, N_EMBD); matrix_init_randn(wte, 0.08); register_matrix(wte);
    wpe = matrix_create(BLOCK_SIZE, N_EMBD); matrix_init_randn(wpe, 0.08); register_matrix(wpe);
    lm_head = matrix_create(vocab_size, N_EMBD); matrix_init_randn(lm_head, 0.08); register_matrix(lm_head);
    
    // Layer 0
    attn_wq = matrix_create(N_EMBD, N_EMBD); matrix_init_randn(attn_wq, 0.08); register_matrix(attn_wq);
    attn_wk = matrix_create(N_EMBD, N_EMBD); matrix_init_randn(attn_wk, 0.08); register_matrix(attn_wk);
    attn_wv = matrix_create(N_EMBD, N_EMBD); matrix_init_randn(attn_wv, 0.08); register_matrix(attn_wv);
    attn_wo = matrix_create(N_EMBD, N_EMBD); matrix_init_randn(attn_wo, 0.08); register_matrix(attn_wo);
    
    mlp_fc1 = matrix_create(4 * N_EMBD, N_EMBD); matrix_init_randn(mlp_fc1, 0.08); register_matrix(mlp_fc1);
    mlp_fc2 = matrix_create(N_EMBD, 4 * N_EMBD); matrix_init_randn(mlp_fc2, 0.08); register_matrix(mlp_fc2);
    
    // Init Cache
    keys_cache = (Value***)kmalloc(BLOCK_SIZE * sizeof(Value**));
    values_cache = (Value***)kmalloc(BLOCK_SIZE * sizeof(Value**));

    vga_printf("Model Params: %d\n", num_params);
}

// "let there be Adam" - Karpathy
static double* m_optim;
static double* v_mom; 

void init_optimizer(void) {
    m_optim = (double*)kmalloc(num_params * sizeof(double));
    v_mom = (double*)kmalloc(num_params * sizeof(double));
    for(int i=0; i<num_params; i++) { m_optim[i]=0; v_mom[i]=0; }
}

void optimizer_step(int t) {
    double alpha = gpt_config.learning_rate;
    double beta1 = gpt_config.beta1;
    double beta2 = gpt_config.beta2;
    double eps = gpt_config.eps;
    
    for(int i=0; i<num_params; i++) {
        Value* p = all_params[i];
        double g = p->grad;
        
        m_optim[i] = beta1 * m_optim[i] + (1.0 - beta1) * g;
        v_mom[i] = beta2 * v_mom[i] + (1.0 - beta2) * g * g;
        
        double m_hat = m_optim[i] / (1.0 - pow(beta1, t+1));
        double v_hat = v_mom[i] / (1.0 - pow(beta2, t+1));
        
        p->data -= alpha * m_hat / (sqrt(v_hat) + eps);
        p->grad = 0; // Zero grad
    }
}

// RMSNorm
static Value** rmsnorm(Value** x, int n) {
    Value* ss = value_create(0.0);
    for(int i=0; i<n; i++) ss = add(ss, mul(x[i], x[i]));
    ss = mul(ss, value_create(1.0/n));
    Value* scale = power(add(ss, value_create(1e-5)), -0.5);
    
    Value** out = (Value**)arena_alloc(n * sizeof(Value*));
    for(int i=0; i<n; i++) out[i] = mul(x[i], scale);
    return out;
}

// Softmax
Value** softmax(Value** logits, int n) {
    Value* max_val = logits[0];
    for(int i=1; i<n; i++) if(logits[i]->data > max_val->data) max_val = logits[i];
    
    Value** exps = (Value**)arena_alloc(n * sizeof(Value*));
    Value* total = value_create(0.0);
    for(int i=0; i<n; i++) {
        // Safe softmax: exp(x - max)
        Value* shifted = add(logits[i], mul(max_val, value_create(-1.0)));
        exps[i] = v_exp(shifted);
        total = add(total, exps[i]);
    }
    
    Value** probs = (Value**)arena_alloc(n * sizeof(Value*));
    Value* inv_total = power(total, -1.0);
    for(int i=0; i<n; i++) probs[i] = mul(exps[i], inv_total);
    return probs;
}

Value** gpt(int token_id, int pos_id) {
    Value** tok_emb = wte.m[token_id];
    Value** pos_emb = wpe.m[pos_id];
    Value** x = (Value**)arena_alloc(N_EMBD * sizeof(Value*));
    for(int i=0; i<N_EMBD; i++) x[i] = add(tok_emb[i], pos_emb[i]);
    
    // Layer 0
    Value** x_residual = x;
    x = rmsnorm(x, N_EMBD);
    
    // Attention
    Value** q = matmul_vec(attn_wq, x);
    Value** k = matmul_vec(attn_wk, x);
    Value** v = matmul_vec(attn_wv, x);
    
    // Store k, v in cache
    keys_cache[pos_id] = k;
    values_cache[pos_id] = v;
    
    // Multi-head attention
    Value** x_attn = (Value**)arena_alloc(N_EMBD * sizeof(Value*));
    for(int h=0; h<N_HEAD; h++) {
        int head_dim = HEAD_DIM;
        int hs = h * head_dim; // start index
        
        // Query for this head
        Value** q_h = &q[hs];
        
        // Attention scores
        // We attend to all past tokens (0..pos_id)
        Value** attn_logits = (Value**)arena_alloc((pos_id+1) * sizeof(Value*));
        
        for(int t=0; t<=pos_id; t++) {
            Value* dot = value_create(0.0);
            Value** k_t = keys_cache[t];
            Value** k_h = &k_t[hs];
            for(int j=0; j<head_dim; j++) {
                dot = add(dot, mul(q_h[j], k_h[j]));
            }
            attn_logits[t] = mul(dot, value_create(1.0 / sqrt(head_dim)));
        }
        
        Value** weights = softmax(attn_logits, pos_id+1);
        
        // Weighted sum of values
        Value** head_out = (Value**)arena_alloc(head_dim * sizeof(Value*));
        for(int j=0; j<head_dim; j++) {
            Value* acc = value_create(0.0);
            for(int t=0; t<=pos_id; t++) {
                Value** v_t = values_cache[t];
                Value** v_h = &v_t[hs];
                acc = add(acc, mul(weights[t], v_h[j]));
            }
            head_out[j] = acc;
        }
        
        // Copy to x_attn
        for(int j=0; j<head_dim; j++) x_attn[hs+j] = head_out[j];
    }
    
    // Projection
    x = matmul_vec(attn_wo, x_attn);
    for(int i=0; i<N_EMBD; i++) x[i] = add(x[i], x_residual[i]);
    
    // MLP block
    x_residual = x;
    x = rmsnorm(x, N_EMBD);
    x = matmul_vec(mlp_fc1, x);
    for(int i=0; i<4*N_EMBD; i++) x[i] = relu(x[i]);
    x = matmul_vec(mlp_fc2, x);
    for(int i=0; i<N_EMBD; i++) x[i] = add(x[i], x_residual[i]);
    
    // LM head
    Value** logits = matmul_vec(lm_head, x);
    return logits;
}

int get_token(char c) {
    for(int i=0; i<vocab_sz; i++) {
        if(unique_chars[i] == c) return i;
    }
    return 0; // Default
}
char get_char(int t) {
    if(t < vocab_sz) return unique_chars[t];
    return '?';
}
