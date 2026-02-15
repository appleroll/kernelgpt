#include "kernelgpt/kernelgpt.h"
#include "kernelgpt/math_utils.h"
#include "kernelgpt/arena.h"
#include "kernelgpt/model.h"
#include "kernelgpt/config.h"
#include "vga_text.h"
#include "names.h"
#include "heap/heap.h"
#include "print/debug.h"

// values taken from microGPT (Andrej Karpathy)
GPTConfig gpt_config = {
    .temperature = 0.5,
    .max_steps = 100,
    .learning_rate = 0.01,
    .beta1 = 0.85,
    .beta2 = 0.99,
    .eps = 1e-8
};

void kernelgpt_main(void) {
    debugf("[KGPT] Starting Training...\n");
    mysrand(1337);
    arena_init();
    
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

    // Traning loop. 
    for(step = 0; step < gpt_config.max_steps; step++) {
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
            
            // Apply temperature
            if (gpt_config.temperature != 1.0) {
                for(int k=0; k<vocab_sz; k++) {
                    logits[k] = mul(logits[k], value_create(1.0/gpt_config.temperature));
                }
            }

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
