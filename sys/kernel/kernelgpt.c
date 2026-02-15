#include "kernelgpt.h"
#include "vga_text.h"
#include "math.h"
#include "heap/heap.h"
#include "print/debug.h"
#include "names.h"

double cos(double x);

// Linear Congruential Generator
static unsigned long int next = 1;
void mysrand(unsigned int seed) { next = seed; }
int myrand(void) {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next / 65536) % 32768;
}

double random_uniform() { return (double)myrand() / 32768.0; }
double random_gauss(double mu, double sigma) {
    // Box-Muller transform
    double u1 = random_uniform();
    double u2 = random_uniform();
    if(u1 < 1e-6) u1 = 1e-6;
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z0 * sigma + mu;
}
double cos(double x) {
    // Taylor series for cos(x) approx
    double term = 1;
    double sum = 1;
    double x2 = x * x;
    for (int i = 1; i <= 10; i++) {
        term *= -x2 / ((2*i) * (2*i-1));
        sum += term;
    }
    return sum;
}

// List implementation
typedef struct {
    void** items;
    int capacity;
    int count;
} List;

List* list_create() {
    List* l = (List*)kmalloc(sizeof(List));
    l->capacity = 16;
    l->count = 0;
    l->items = (void**)kmalloc(sizeof(void*) * l->capacity);
    return l;
}
void list_append(List* l, void* item) {
    if (l->count == l->capacity) {
        l->capacity *= 2;
        l->items = (void**)krealloc(l->items, sizeof(void*) * l->capacity);
    }
    l->items[l->count++] = item;
}
void list_free(List* l) {
    kfree(l->items);
    kfree(l);
}

// Arena allocator for autograd graph nodes - we will reset it after each training iteration
#define ARENA_SIZE (8 * 1024 * 1024) // 8MB arena
static char* arena_buffer = NULL;
static size_t arena_offset = 0;

void arena_init() {

    arena_buffer = (char*)kmalloc(ARENA_SIZE);
    if (!arena_buffer) {
        vga_puts("PANIC: Failed to allocate Arena!\n");
        while(1);
    }
    arena_offset = 0;
    vga_puts("Arena allocated.\n");
}
void arena_reset() {
    arena_offset = 0;
}
void* arena_alloc(size_t size) {
    // Align to 8 bytes
    if (arena_offset % 8 != 0) arena_offset += (8 - (arena_offset % 8));
    if (arena_offset + size > ARENA_SIZE) {
        vga_puts("PANIC: Arena Out of Memory!\n");
        while(1);
    }
    void* ptr = &arena_buffer[arena_offset];
    arena_offset += size;
    return ptr;
}

// --- Autograd Engine ---

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

// Matrix 
// I don't think MicroGPT uses Matrixes, but I think matrixes are pretty cool and a good thing to have.
typedef struct {
    int rows;
    int cols;
    Value*** m; // m[rows][cols]
} Matrix;

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

// Init random matrix (Parameters)
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

#define BLOCK_SIZE 16
#define N_EMBD 16
#define N_HEAD 4
#define N_LAYER 1
#define HEAD_DIM (N_EMBD / N_HEAD)
#define VOCAB_SIZE 65 // will calculate

Matrix wte;
Matrix wpe;
Matrix lm_head;
Matrix attn_wq, attn_wk, attn_wv, attn_wo;
Matrix mlp_fc1, mlp_fc2;

// To collect all params for optimizer
#define MAX_PARAMS 10000
Value* all_params[MAX_PARAMS];
int num_params = 0;

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
    
    vga_printf("Model Params: %d\n", num_params);
}

// RMSNorm
Value** rmsnorm(Value** x, int n) {
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

Value*** keys_cache; // [BLOCK_SIZE][N_EMBD]
Value*** values_cache;

void init_cache() {
    keys_cache = (Value***)kmalloc(BLOCK_SIZE * sizeof(Value**));
    values_cache = (Value***)kmalloc(BLOCK_SIZE * sizeof(Value**));
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

// "let there be Adam" - Karpathy
double* m;
double* v_mom; 
void init_optimizer() {
    m = (double*)kmalloc(num_params * sizeof(double));
    v_mom = (double*)kmalloc(num_params * sizeof(double));
    for(int i=0; i<num_params; i++) { m[i]=0; v_mom[i]=0; }
}

void optimizer_step(int t) {
    double alpha = 0.01;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    
    for(int i=0; i<num_params; i++) {
        Value* p = all_params[i];
        double g = p->grad;
        
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v_mom[i] = beta2 * v_mom[i] + (1.0 - beta2) * g * g;
        
        double m_hat = m[i] / (1.0 - pow(beta1, t+1));
        double v_hat = v_mom[i] / (1.0 - pow(beta2, t+1));
        
        p->data -= alpha * m_hat / (sqrt(v_hat) + eps);
        p->grad = 0; // Zero grad
    }
}

// Data Processing
int* tokens;
int num_tokens;
char* unique_chars;
int vocab_sz;

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

void kernelgpt_main() {
    debugf("KernelGPT: Starting Training...\n");
    mysrand(1337);
    arena_init();
    init_cache(); // allocate pointers
    
    vga_puts("Processing Dataset...\n");

    // Build vocab
    char uchar_temp[256] = {0};
    for(unsigned int i=0; i<names_txt_len; i++) {
        uchar_temp[(unsigned char)names_txt[i]] = 1;
    }
    // Collect unique
    int u_count = 0;

    for(int c=0; c<256; c++) {
        if(uchar_temp[c] && c != '\n') u_count++;
    }
    vocab_sz = u_count + 1; // +1 for BOS
    int BOS = u_count;
    
    unique_chars = (char*)kmalloc(vocab_sz);
    int idx=0;
    for(int c=0; c<256; c++) {
        if(uchar_temp[c] && c != '\n') unique_chars[idx++] = (char)c;
    }
    
    vga_printf("Vocab Size: %d\n", vocab_sz);
    
    init_model(vocab_sz);
    init_optimizer();
    
    vga_puts("Training...\n");
    
    int doc_start = 0;
    int step = 0;
    int total_docs = 0;

    // Traning loop. 
    // @todo - increase steps?
    for(step = 0; step < 50; step++) {
        arena_reset();
        
        int doc_end = doc_start;
        while(doc_end < names_txt_len && names_txt[doc_end] != '\n') doc_end++;
        
        int doc_len = doc_end - doc_start;
        if(doc_len <= 0) { // skip empty lines
             doc_start = doc_end + 1;
             if(doc_start >= names_txt_len) doc_start = 0;
             step--; continue; 
        }
        
        int seq_len = doc_len + 2; // BOS ... BOS
        int* seq = (int*)arena_alloc(seq_len * sizeof(int));
        seq[0] = BOS;
        for(int i=0; i<doc_len; i++) seq[i+1] = get_token(names_txt[doc_start+i]);
        seq[doc_len+1] = BOS;
        
        doc_start = doc_end + 1;
        if(doc_start >= names_txt_len) doc_start = 0;
        
        // Forward
        int n = seq_len - 1; 
        if(n > BLOCK_SIZE) n = BLOCK_SIZE; // Truncate
        
        Value* total_loss = value_create(0.0);
        
        for(int pos=0; pos<n; pos++) {
            int target = seq[pos+1];
            Value** logits = gpt(seq[pos], pos);
            
            // Cross Entropy Loss
            Value** prob = softmax(logits, vocab_sz);
            Value* neg_log_prob = mul(v_log(prob[target]), value_create(-1.0));
            total_loss = add(total_loss, neg_log_prob);
        }
        
        total_loss = mul(total_loss, value_create(1.0/n));
        
        backward(total_loss);
        optimizer_step(step);
        
        if(step % 1 == 0) {
            vga_printf("Step %d | Loss: %f\n", step, total_loss->data);
        }
    }
    
    vga_puts("\nTraining Complete! Generating Names:\n");
    
    // Inference
    for(int i=0; i<10; i++) {
        arena_reset();
        int token = BOS;
        vga_printf("Name %d: ", i+1);
        for(int pos=0; pos<BLOCK_SIZE; pos++) {
            
            Value** logits = gpt(token, pos);
            Value** probs = softmax(logits, vocab_sz);
            double r = random_uniform();
            double cdf = 0.0;
            int next_token = 0;
            for(int j=0; j<vocab_sz; j++) {
                cdf += probs[j]->data;
                if(r < cdf) {
                    next_token = j;
                    break;
                }
            }
            
            token = next_token;
            if(token == BOS) break;
            vga_putc(get_char(token));
        }
        vga_putc('\n');
    }
}
